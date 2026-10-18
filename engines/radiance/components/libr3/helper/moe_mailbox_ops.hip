#include <cstdio>
#include "helper/moe_remote_hook.h"
#include "helper/moe_remote_shm.h"
#include "rad_abi.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
using namespace r3::moe_shm;

#define NELEM(a) ((int)(sizeof(a) / sizeof((a)[0])))

static const RadParamSpec kPublishParams[] = {
    {"layer", RAD_P_INT, RAD_REQUIRED},
    {"M", RAD_P_INT, RAD_REQUIRED},
    {"n", RAD_P_INT, RAD_REQUIRED},
    {"top_k", RAD_P_INT, RAD_REQUIRED},
    {"expert_base", RAD_P_INT, RAD_REQUIRED},
    {"n_expert", RAD_P_INT, RAD_REQUIRED},
};
static const RadOperandSpec kPublishOperands[] = {
    {"ids", RAD_OPD_INOUT, 0},
    {"ew", RAD_OPD_IN, 0},
    {"x", RAD_OPD_IN, 0},
    {"aq", RAD_OPD_IN, 0},
    {"as", RAD_OPD_IN, 0},
    {"aq_base", RAD_OPD_IN, 0},
};
static const RadParamSpec kAddParams[] = {
    {"M", RAD_P_INT, RAD_REQUIRED},
    {"n", RAD_P_INT, RAD_REQUIRED},
    {"top_k", RAD_P_INT, RAD_REQUIRED},
};
static const RadOperandSpec kAddOperands[] = {
    {"sorted", RAD_OPD_IN, 0},
    {"down", RAD_OPD_IN, 0},
    {"ew", RAD_OPD_IN, 0},
    {"shared", RAD_OPD_IN, 1},
    {"shared_gate", RAD_OPD_IN, 1},
    {"out", RAD_OPD_OUT, 0},
};
static const RadParamSpec kMaskRefParams[] = {
    {"M", RAD_P_INT, RAD_REQUIRED},
    {"n", RAD_P_INT, RAD_REQUIRED},
    {"top_k", RAD_P_INT, RAD_REQUIRED},
    {"restore", RAD_P_INT, RAD_REQUIRED},
};
static const RadOperandSpec kMaskRefOperands[] = {
    {"sorted", RAD_OPD_IN, 0},
    {"down", RAD_OPD_INOUT, 0},
};
static const RadParamSpec kShadowParams[] = {
    {"M", RAD_P_INT, RAD_REQUIRED},
    {"n", RAD_P_INT, RAD_REQUIRED},
    {"top_k", RAD_P_INT, RAD_REQUIRED},
};
static const RadOperandSpec kShadowOperands[] = {
    {"sorted", RAD_OPD_IN, 0},
    {"down", RAD_OPD_INOUT, 0},
    {"ew", RAD_OPD_IN, 0},
};
static const RadOpSchema kSchemas[] = {
    {"moe_mailbox_publish", kPublishParams, NELEM(kPublishParams),
     kPublishOperands, NELEM(kPublishOperands),
     "publish helper-owned MoE routes and activations to the mapped GPU mailbox"},
    {"moe_mailbox_add", kAddParams, NELEM(kAddParams),
     kAddOperands, NELEM(kAddOperands),
     "merge local expert rows (slot order, f32) with the helper per-token weighted row and the shared expert"},
    {"moe_mailbox_mask_ref", kMaskRefParams, NELEM(kMaskRefParams),
     kMaskRefOperands, NELEM(kMaskRefOperands),
     "capture or restore engine down rows for the mask-only diagnostic"},
    {"moe_mailbox_shadow", kShadowParams, NELEM(kShadowParams),
     kShadowOperands, NELEM(kShadowOperands),
     "compare the helper per-token weighted row against the engine own rows without changing model output"},
};

__device__ __forceinline__ void store_sys_rel(uint32_t* p, uint32_t v) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ uint32_t load_sys_acq(const uint32_t* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ bool owner_has(const uint8_t* bits, uint32_t layer, uint32_t expert) {
    const uint64_t bit = (uint64_t)layer * kExperts + expert;
    return (bits[bit >> 3] & (uint8_t)(1u << (bit & 7u))) != 0;
}

/* Which rank slots this step publishes to: a two-rank engine uses its own slot (helper 0 serves both); a
 * one-rank engine with two helpers uses slot h for helper h, the routes split by which helper owns the expert. */
__device__ __forceinline__ uint32_t step_slot(uint32_t k, uint32_t rank, uint32_t helpers) {
    return helpers > 1 ? k : rank;
}

/* The step's exact activation operand (E4M3 codes + f32 group scales, and the bf16 rows for the host
 * reference) into each staging slot the step publishes to, across the whole grid; the publish kernel that
 * follows on the same stream signals the helpers only after it. Each helper has finished the slot's previous
 * request (the merge waited on it), so the slots are free. Copied 16 bytes a thread. */
__global__ void mailbox_stage_kernel(
    uint8_t* base, uint32_t rank, uint32_t helpers, const uint8_t* aq, const float* as, const uint16_t* x,
    uint32_t rows, uint32_t hidden, uint32_t stage_bf16) {
    Header* h = reinterpret_cast<Header*>(base);
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    const uint64_t t0 = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    /* One staging copy: a lone helper reads its rank slot, two helpers share slot 0 (aliased by helper 0). */
    for (uint32_t k = 0; k < 1u; ++k) {
        const uint32_t slot = step_slot(k, rank, helpers);
        uint8_t* staged_q = base + h->activation_offset[slot] + kActivationBf16Bytes;
        float* staged_s = reinterpret_cast<float*>(
            base + h->activation_offset[slot] + kActivationBf16Bytes + kActivationQBytes);
        const uint64_t q16 = (uint64_t)rows * hidden / 16u;
        for (uint64_t i = t0; i < q16; i += stride)
            reinterpret_cast<uint4*>(staged_q)[i] = reinterpret_cast<const uint4*>(aq)[i];
        const uint64_t s_total = (uint64_t)rows * (hidden / kActGroup);
        for (uint64_t i = t0; i < s_total; i += stride) staged_s[i] = as[i];
        if (stage_bf16) {
            uint16_t* staged_x = reinterpret_cast<uint16_t*>(base + h->activation_offset[slot]);
            const uint64_t x8 = (uint64_t)rows * hidden / 8u;
            for (uint64_t i = t0; i < x8; i += stride)
                reinterpret_cast<uint4*>(staged_x)[i] = reinterpret_cast<const uint4*>(x)[i];
        }
    }
}

__global__ void mailbox_publish_kernel(
    uint8_t* base, const uint8_t* owner, uint32_t* sequence, uint32_t* published, uint32_t rank,
    uint32_t helpers, int32_t* ids, const uint16_t* ew, uint32_t* heat,
    const uint8_t* aq, const uint8_t* aq_base,
    uint32_t layer, uint32_t rows, uint32_t hidden, uint32_t top_k,
    int32_t expert_base, uint32_t n_expert, uint32_t max_rows,
    uint32_t shadow, uint32_t world) {
    __shared__ uint32_t seq;
    __shared__ uint32_t nr;

    Header* h = reinterpret_cast<Header*>(base);
    if (threadIdx.x < kMaxRanks) published[threadIdx.x] = 0;
    __syncthreads();
    if (rows == 0 || rows > max_rows || rows > kMaxRows ||
        layer >= h->layer_count || !load_sys_acq(&h->layer_map_ready[layer])) return;

    const int32_t* slot_map = reinterpret_cast<const int32_t*>(
        base + h->layer_map_offset) + (uint64_t)layer * kExperts;
    const uint32_t large = rows > h->small_rows ? 1u : 0u;
    const uint32_t scale_cols = hidden / kActGroup;
    const uintptr_t qptr = reinterpret_cast<uintptr_t>(aq);
    const uintptr_t qbase = reinterpret_cast<uintptr_t>(aq_base);
    if (qptr < qbase || ((qptr - qbase) % hidden) != 0) __builtin_trap();

    for (uint32_t k = 0; k < helpers; ++k) {
        const uint32_t slot = step_slot(k, rank, helpers);
        const uint8_t* own = owner + (size_t)k * h->owner_slot_bytes;
        if (threadIdx.x == 0) nr = 0;
        __syncthreads();
        Route* route = reinterpret_cast<Route*>(base + h->route_offset[slot]);
        for (uint32_t i = threadIdx.x; i < rows * top_k; i += blockDim.x) {
            const int32_t le = ids[i] - expert_base;
            const bool in_range = le >= 0 && (uint32_t)le < n_expert && (uint32_t)le < kExperts;
            /* R3_MOE_HELPER_HISTOGRAM: every route this step makes, owned or not, counted once (first slot). */
            if (heat && k == 0 && in_range) atomicAdd(heat + (size_t)layer * kExperts + (uint32_t)le, 1u);
            if (!in_range || slot_map[(uint32_t)le] < 0 || !owner_has(own, layer, (uint32_t)le))
                continue;
            const uint32_t p = atomicAdd(&nr, 1u);
            route[p].token = (int32_t)(i / top_k);
            route[p].expert = le;
            route[p].weight_bf16 = ew[i];
            route[p].slot = (uint16_t)(i % top_k);
            if (!shadow) ids[i] = -1;
        }
        __syncthreads();
        if (nr == 0) continue;

        const uint32_t lane = lane_of(slot, large);
        if (threadIdx.x == 0) {
            seq = atomicAdd(sequence + slot * 2u + large, 1u) + 1u;
            if (seq == 0) __builtin_trap();
        }
        __syncthreads();
        Request* ring = reinterpret_cast<Request*>(base + h->ring_offset);
        Request* q = ring + lane * kLaneSlots + ((seq - 1u) % kLaneSlots);
        if (threadIdx.x == 0) {
            q->completion_sequence = 0;
            q->status = 0;
            q->layer = layer;
            q->rank = slot;
            q->rank_count = world;
            q->rows = rows;
            q->hidden = hidden;
            q->route_count = nr;
            q->row_base = (uint32_t)((qptr - qbase) / hidden);
            q->route_offset = h->route_offset[slot];
            q->activation_offset = h->activation_offset[slot];
            q->activation_q_offset = h->activation_offset[slot] + kActivationBf16Bytes;
            q->activation_s_offset =
                h->activation_offset[slot] + kActivationBf16Bytes + kActivationQBytes;
            q->activation_stride = hidden * (uint32_t)sizeof(uint16_t);
            q->activation_q_stride = hidden;
            q->activation_s_stride = scale_cols * (uint32_t)sizeof(float);
            for (uint32_t r = 0; r < kMaxRanks; ++r) {
                q->result_offset[r] = h->result_offset[slot * kMaxRanks + r];
                q->result_stride[r] = hidden * (uint32_t)sizeof(uint16_t);
            }
            q->sequence = seq;
            q->trace[0] = wall_clock64();
            q->trace[1] = 0;
            q->trace[2] = 0;
            q->trace[3] = 0;
            q->trace[4] = 0;
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            store_sys_rel(&h->lane[lane].request_sequence, seq);
            published[slot] = 1u + large;   /* 0: nothing published to this slot; else 1 + the lane class */
        }
        __syncthreads();
    }
}

__device__ __forceinline__ float bf16_to_f32(uint16_t v) {
    return __uint_as_float((uint32_t)v << 16);
}

__device__ __forceinline__ uint16_t f32_to_bf16_rn(float v) {
    uint32_t u = __float_as_uint(v);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

/* The request a slot carries this step, or null when the step published nothing to it. */
__device__ __forceinline__ const Request* slot_request(
    uint8_t* base, const uint32_t* sequence, const uint32_t* published, uint32_t slot) {
    if (!published[slot]) return nullptr;
    Header* h = reinterpret_cast<Header*>(base);
    const uint32_t large = published[slot] - 1u;
    const uint32_t seq = sequence[slot * 2u + large];
    const Request* ring = reinterpret_cast<const Request*>(base + h->ring_offset);
    return ring + lane_of(slot, large) * kLaneSlots + ((seq - 1u) % kLaneSlots);
}

__global__ void mailbox_add_wait_kernel(
    uint8_t* base, const uint32_t* sequence, const uint32_t* published, uint32_t rank, uint32_t helpers) {
    if (blockIdx.x || threadIdx.x) return;
    Header* h = reinterpret_cast<Header*>(base);
    const uint64_t entered = wall_clock64();   /* R3_MOE_HELPER_TRACE: engine clock */
    Request* traced[kMaxRanks] = {};
    uint64_t seen_at[kMaxRanks] = {};
    for (uint32_t k = 0; k < helpers && k < kMaxRanks; ++k) {
        const uint32_t slot = step_slot(k, rank, helpers);
        if (!published[slot]) continue;
        const uint32_t large = published[slot] - 1u;
        const uint32_t seq = sequence[slot * 2u + large];
        const uint32_t lane = lane_of(slot, large);
        unsigned long long spins = 0;
        /* Exactly this request's generation: one request is in flight a lane, so the helper's done word
         * reaches seq and no further before the engine's next publish. */
        while (load_sys_acq(&h->lane[lane].done_sequence) != seq) {
            if (load_sys_acq(&h->helper_fault)) __builtin_trap();   /* a helper stopped: fail now, not hang */
            __builtin_amdgcn_s_sleep(2);
            if (++spins > 4000000000ULL) __builtin_trap();
        }
        /* Keep the completion time in a register: a store to mapped host memory inside the loop would be counted in the
         * second helper's wait. Both are written once the last wait is over. */
        seen_at[k] = wall_clock64();
        traced[k] = reinterpret_cast<Request*>(base + h->ring_offset) + lane * kLaneSlots + ((seq - 1u) % kLaneSlots);
    }
    for (uint32_t k = 0; k < helpers && k < kMaxRanks; ++k) {
        if (!traced[k]) continue;
        traced[k]->trace[1] = entered;
        traced[k]->trace[2] = seen_at[k];
    }
}

__global__ void mailbox_mask_ref_kernel(
    uint16_t* mask_ref, const int32_t* sorted, uint16_t* down,
    uint32_t slots, uint32_t hidden, uint32_t restore) {
    const uint32_t row = blockIdx.x;
    if (row >= slots) return;
    const int32_t flat = sorted[row];
    if (flat < 0 || (uint32_t)flat >= slots) return;
    for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        uint16_t* ref = mask_ref + (uint64_t)(uint32_t)flat * hidden + col;
        uint16_t* local = down + (uint64_t)row * hidden + col;
        if (restore) *local = *ref;
        else *ref = *local;
    }
}

/* Per-step maps the merge and shadow kernels read instead of scanning the step: rmask[flat] is 1 + the
 * slot index k of the helper that computed the slot (0 when none; checked against the router's weight),
 * inv[flat] is the engine's sorted row for a slot it computed locally (-1 otherwise). Linear in the step,
 * so a 4096-row prefill step costs the same per row as a decode step. */
__global__ void mailbox_route_mark_kernel(
    uint8_t* base, const uint32_t* sequence, const uint32_t* published, uint32_t rank, uint32_t helpers,
    const uint16_t* ew, uint32_t* rmask, uint32_t rows, uint32_t hidden, uint32_t top_k) {
    Header* h = reinterpret_cast<Header*>(base);
    for (uint32_t k = 0; k < helpers; ++k) {
        const uint32_t slot = step_slot(k, rank, helpers);
        const Request* q = slot_request(base, sequence, published, slot);
        if (!q) continue;
        const uint32_t large = published[slot] - 1u;
        if (load_sys_acq(&h->lane[lane_of(slot, large)].done_sequence) != sequence[slot * 2u + large])
            __builtin_trap();
        if (q->sequence != sequence[slot * 2u + large] || q->rank != slot || q->rows != rows ||
            q->hidden != hidden || q->status != 0 || q->route_count == 0 || q->route_count > rows * top_k)
            __builtin_trap();
        const Route* route = reinterpret_cast<const Route*>(base + q->route_offset);
        for (uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < q->route_count;
             j += gridDim.x * blockDim.x) {
            const Route r = route[j];
            if (r.token < 0 || (uint32_t)r.token >= rows || r.slot >= top_k) __builtin_trap();
            const uint32_t flat = (uint32_t)r.token * top_k + r.slot;
            if (r.weight_bf16 != ew[flat]) __builtin_trap();
            if (atomicCAS(rmask + flat, 0u, k + 1u) != 0u) __builtin_trap();   /* two routes for one slot */
        }
    }
}

__global__ void mailbox_inverse_kernel(const int32_t* sorted, int32_t* inv, uint32_t slots) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= slots) return;
    const int32_t flat = sorted[i];
    if (flat >= 0 && (uint32_t)flat < slots) inv[flat] = (int32_t)i;
}

/* Each helper returns ONE row a token: the router-weighted sum of that token's experts it computed, in slot
 * order with an f32 accumulator, narrowed once to bf16. The merge adds each helper's row to the token's
 * locally computed slots (f32, slot order) and the shared expert. Mask-only mode 1 merges the engine's own
 * captured rows for the helper's slots instead (the exact control); mode 2 merges the helpers' rows on the
 * late-publish schedule. */
__global__ void mailbox_add_rows_kernel(
    uint8_t* base, const uint32_t* published, uint32_t rank, uint32_t helpers,
    const int32_t* inv, const uint32_t* rmask, const uint16_t* down, const uint16_t* ew,
    const uint16_t* shared, const uint16_t* shared_gate, uint16_t* out,
    const uint16_t* mask_ref, uint32_t mask_only,
    uint32_t rows, uint32_t hidden, uint32_t top_k) {
    const uint32_t token = blockIdx.x;
    if (token >= rows) return;
    Header* h = reinterpret_cast<Header*>(base);
    __shared__ int32_t local_index[kTopK];
    __shared__ uint8_t remote_slot[kTopK];
    __shared__ uint32_t from[kMaxRanks];
    if (threadIdx.x == 0) {
        bool live = false;
        for (uint32_t k = 0; k < helpers; ++k) live |= published[step_slot(k, rank, helpers)] != 0u;
        for (uint32_t k = 0; k < kMaxRanks; ++k) from[k] = 0;
        for (uint32_t slot = 0; slot < top_k; ++slot) {
            const uint32_t flat = token * top_k + slot;
            local_index[slot] = inv[flat];
            const uint32_t m = live ? rmask[flat] : 0u;
            remote_slot[slot] = (uint8_t)m;
            if (m) from[m - 1u] = 1;
            if (mask_only && m && local_index[slot] >= 0) __builtin_trap();
        }
    }
    __syncthreads();

    float gate = 0.0f;
    if (shared) {
        const float v = bf16_to_f32(shared_gate[token]);
        gate = 1.0f / (1.0f + __expf(-v));
    }
    for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t slot = 0; slot < top_k; ++slot) {
            const uint32_t flat = token * top_k + slot;
            const uint16_t* src = nullptr;
            if (remote_slot[slot]) {
                if (mask_only == 1u) src = mask_ref + (uint64_t)flat * hidden;
            } else if (local_index[slot] >= 0) {
                src = down + (uint64_t)(uint32_t)local_index[slot] * hidden;
            }
            if (src) acc += bf16_to_f32(ew[flat]) * bf16_to_f32(src[col]);
        }
        /* Host-mapped and rewritten every request by the helpers: read past the caches. */
        if (mask_only != 1u)
            for (uint32_t k = 0; k < helpers; ++k) {
                if (!from[k]) continue;
                const uint32_t slot = step_slot(k, rank, helpers);
                const uint16_t* result = reinterpret_cast<const uint16_t*>(
                    base + h->result_offset[slot * kMaxRanks + slot]) + (uint64_t)token * hidden;
                acc += bf16_to_f32(__builtin_nontemporal_load(result + col));
            }
        const uint16_t gathered = f32_to_bf16_rn(acc);
        if (!shared) {
            out[(uint64_t)token * hidden + col] = gathered;
            continue;
        }
        const uint16_t part = f32_to_bf16_rn(
            bf16_to_f32(shared[(uint64_t)token * hidden + col]) * gate);
        out[(uint64_t)token * hidden + col] =
            f32_to_bf16_rn(bf16_to_f32(part) + bf16_to_f32(gathered));
    }
}

/* Shadow: the engine computed every slot locally; compare, per token with helper routes, the sum of the
 * helpers' returned rows with the same weighted sum over the engine's own rows for those slots. */
__global__ void mailbox_shadow_rows_kernel(
    uint8_t* base, const uint32_t* sequence, const uint32_t* published, uint32_t rank, uint32_t helpers,
    const int32_t* inv, const uint32_t* rmask, const uint16_t* down, const uint16_t* ew,
    uint32_t rows, uint32_t hidden, uint32_t top_k,
    R3MoeShadowLayerStats* stats, R3MoeShadowFirst* first) {
    const uint32_t token = blockIdx.x;
    if (token >= rows) return;
    Header* h = reinterpret_cast<Header*>(base);
    const Request* q = nullptr;
    for (uint32_t k = 0; k < helpers && !q; ++k) q = slot_request(base, sequence, published, step_slot(k, rank, helpers));
    if (!q || q->layer >= kMaxLayers) return;
    __shared__ uint32_t any;
    __shared__ uint32_t missing;
    __shared__ uint32_t from[kMaxRanks];
    if (threadIdx.x == 0) {
        any = 0; missing = 0;
        for (uint32_t k = 0; k < kMaxRanks; ++k) from[k] = 0;
        for (uint32_t slot = 0; slot < top_k; ++slot) {
            const uint32_t flat = token * top_k + slot;
            if (!rmask[flat]) continue;
            any = 1;
            from[rmask[flat] - 1u] = 1;
            if (inv[flat] < 0) missing = 1;
        }
    }
    __syncthreads();
    if (!any) return;
    R3MoeShadowLayerStats* st = stats + q->layer;
    if (missing) {
        if (threadIdx.x == 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&st->rows), 1ull);
            atomicAdd(&st->bad_rows, 1u);
        }
        return;
    }
    __shared__ float sa[256], sd2[256], sr2[256], sm[256];
    __shared__ float lref[4], lhelp[4];
    float a = 0.0f, d2 = 0.0f, r2 = 0.0f, m = 0.0f;
    for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        float ref = 0.0f;
        for (uint32_t slot = 0; slot < top_k; ++slot) {
            const uint32_t flat = token * top_k + slot;
            if (!rmask[flat]) continue;
            ref += bf16_to_f32(ew[flat]) * bf16_to_f32(down[(uint64_t)(uint32_t)inv[flat] * hidden + col]);
        }
        ref = bf16_to_f32(f32_to_bf16_rn(ref));
        float hv = 0.0f;
        for (uint32_t k = 0; k < helpers; ++k) {
            if (!from[k]) continue;
            const uint32_t slot = step_slot(k, rank, helpers);
            hv += bf16_to_f32(reinterpret_cast<const uint16_t*>(
                base + h->result_offset[slot * kMaxRanks + slot])[(uint64_t)token * hidden + col]);
        }
        if (col < 4) { lref[col] = ref; lhelp[col] = hv; }
        const float d = ref - hv;
        a += fabsf(d); d2 += d * d; r2 += ref * ref; m = fmaxf(m, fabsf(d));
    }
    sa[threadIdx.x] = a; sd2[threadIdx.x] = d2; sr2[threadIdx.x] = r2; sm[threadIdx.x] = m;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            sa[threadIdx.x] += sa[threadIdx.x + stride];
            sd2[threadIdx.x] += sd2[threadIdx.x + stride];
            sr2[threadIdx.x] += sr2[threadIdx.x + stride];
            sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float rel = sqrtf(sd2[0] / fmaxf(sr2[0], 1.0e-30f));
        atomicAdd(reinterpret_cast<unsigned long long*>(&st->rows), 1ull);
        atomicAdd(reinterpret_cast<unsigned long long*>(&st->elems), (unsigned long long)hidden);
        atomicAdd(&st->sum_abs, sa[0]);
        atomicAdd(&st->diff2, sd2[0]);
        atomicAdd(&st->ref2, sr2[0]);
        atomicMax(&st->max_abs_bits, __float_as_uint(sm[0]));
        if (!isfinite(rel) || rel >= 2.0e-3f) {
            atomicAdd(&st->bad_rows, 1u);
            R3MoeShadowFirst* f = first + q->layer;
            if (atomicCAS(&f->claimed, 0u, 1u) == 0u) {
                f->layer = q->layer; f->rank = q->rank; f->row_base = q->row_base;
                f->token = (int32_t)token; f->slot = 0xffffu; f->expert = -1;
                f->rel_l2 = rel; f->max_abs = sm[0];
                for (uint32_t i = 0; i < 4; ++i) { f->local[i] = lref[i]; f->helper[i] = lhelp[i]; }
            }
        }
    }
}

__global__ void mailbox_add_finish_kernel(uint32_t* published) {
    if (!blockIdx.x && threadIdx.x < kMaxRanks) published[threadIdx.x] = 0;
}

long long geti(const RadArgs* a, const char* key, long long bad = -1) {
    long long v = bad;
    return rad_args_geti(a, key, &v) ? v : bad;
}

int mailbox_init(const RadParam*, int, int rank, int world_size, void** out) {
    if (!out) return RAD_E_INVAL;
    R3MoeMailboxDevice* d = nullptr;
    const int rc = r3_moe_mailbox_device(rank, world_size, &d);
    if (rc != RAD_OK) return rc;
    *out = d;
    return RAD_OK;
}

int publish_launch(const RadArgs* a, RadStream stream) {
    if (!a || !a->instance || !stream || a->n_t != 6) return RAD_E_INVAL;
    auto* d = static_cast<R3MoeMailboxDevice*>(a->instance);
    const long long layer = geti(a, "layer");
    const long long hidden = geti(a, "n");
    const long long top_k = geti(a, "top_k");
    const long long expert_base = geti(a, "expert_base");
    const long long n_expert = geti(a, "n_expert");
    const long long rows = a->t[3].rank > 0 ? a->t[3].shape[0] : 0;
    if (layer < 0 || layer >= (long long)kMaxLayers || rows <= 0 || rows > (long long)kMaxRows ||
        hidden != (long long)kHidden || top_k != (long long)kTopK ||
        expert_base < 0 || n_expert <= 0 || n_expert > (long long)kExperts)
        return RAD_E_SHAPE;
    for (int i = 0; i < 6; ++i) if (!a->t[i].data) return RAD_E_SHAPE;
    if (a->t[0].rank < 2 || a->t[1].rank < 2 || a->t[2].rank < 2 ||
        a->t[3].rank < 2 || a->t[4].rank < 2 || a->t[5].rank < 2 ||
        a->t[0].shape[0] != rows || a->t[1].shape[0] != rows ||
        a->t[2].shape[0] != rows || a->t[4].shape[0] != rows ||
        a->t[0].shape[1] != top_k || a->t[1].shape[1] != top_k ||
        a->t[2].shape[1] != hidden || a->t[3].shape[1] != hidden ||
        a->t[4].shape[1] != hidden / kActGroup || a->t[5].shape[1] != hidden ||
        a->t[0].dtype != RAD_I32 || a->t[1].dtype != RAD_BF16 ||
        a->t[2].dtype != RAD_BF16 || a->t[3].dtype != RAD_F8E4M3 ||
        a->t[4].dtype != RAD_F32 || a->t[5].dtype != RAD_F8E4M3)
        return RAD_E_SHAPE;
    for (int i = 0; i < 6; ++i)
        if (!rad_tensor_is_contiguous(&a->t[i])) return RAD_E_SHAPE;

    const uintptr_t q = reinterpret_cast<uintptr_t>(a->t[3].data);
    const uintptr_t qb = reinterpret_cast<uintptr_t>(a->t[5].data);
    if (q < qb || ((q - qb) % (uintptr_t)hidden) != 0 ||
        (long long)((q - qb) / (uintptr_t)hidden) + rows > a->t[5].shape[0])
        return RAD_E_SHAPE;

    /* kHidden is a multiple of 16, so the 16-byte copies cover every row exactly. */
    const uint32_t blocks16 = ((uint32_t)rows * (uint32_t)hidden / 16u + 255u) / 256u;
    hipLaunchKernelGGL(mailbox_stage_kernel, dim3(blocks16 > 512u ? 512u : blocks16),
                       dim3(256), 0, (hipStream_t)stream, static_cast<uint8_t*>(d->mailbox), d->rank,
                       d->helpers, static_cast<const uint8_t*>(a->t[3].data),
                       static_cast<const float*>(a->t[4].data),
                       static_cast<const uint16_t*>(a->t[2].data), (uint32_t)rows, (uint32_t)hidden,
                       d->cpu_ref);
    hipLaunchKernelGGL(mailbox_publish_kernel, dim3(1), dim3(256), 0, (hipStream_t)stream,
                       static_cast<uint8_t*>(d->mailbox), d->owner, d->sequence, d->published, d->rank,
                       d->helpers, static_cast<int32_t*>(a->t[0].data),
                       static_cast<const uint16_t*>(a->t[1].data), d->heat,
                       static_cast<const uint8_t*>(a->t[3].data),
                       static_cast<const uint8_t*>(a->t[5].data),
                       (uint32_t)layer, (uint32_t)rows, (uint32_t)hidden, (uint32_t)top_k,
                       (int32_t)expert_base, (uint32_t)n_expert, d->max_rows,
                       d->shadow, d->world);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int mask_ref_launch(const RadArgs* a, RadStream stream) {
    if (!a || !a->instance || !stream || a->n_t != 2) return RAD_E_INVAL;
    auto* d = static_cast<R3MoeMailboxDevice*>(a->instance);
    if (!d->mask_only || !d->mask_ref) return RAD_E_STATE;
    const long long hidden = geti(a, "n");
    const long long top_k = geti(a, "top_k");
    const long long restore = geti(a, "restore");
    const long long slots = a->t[1].rank > 0 ? a->t[1].shape[0] : 0;
    if (hidden != (long long)kHidden || top_k != (long long)kTopK ||
        (restore != 0 && restore != 1) || slots <= 0 || slots % top_k != 0 ||
        slots / top_k > (long long)d->max_rows ||
        !a->t[0].data || !a->t[1].data || a->t[0].dtype != RAD_I32 ||
        a->t[1].dtype != RAD_BF16 || a->t[0].rank < 1 || a->t[1].rank < 2 ||
        a->t[0].shape[0] < slots || a->t[1].shape[1] != hidden ||
        !rad_tensor_is_contiguous(&a->t[0]) || !rad_tensor_is_contiguous(&a->t[1]))
        return RAD_E_SHAPE;

    if (!restore)
        (void)hipMemsetAsync(d->mask_ref, 0,
            (size_t)slots * (size_t)hidden * sizeof(uint16_t), (hipStream_t)stream);
    hipLaunchKernelGGL(mailbox_mask_ref_kernel, dim3((uint32_t)slots), dim3(256), 0,
                       (hipStream_t)stream, d->mask_ref,
                       static_cast<const int32_t*>(a->t[0].data),
                       static_cast<uint16_t*>(a->t[1].data),
                       (uint32_t)slots, (uint32_t)hidden, (uint32_t)restore);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* Wait for the helper, then build the step's slot maps (see mailbox_route_mark_kernel). */
int launch_maps(R3MoeMailboxDevice* d, RadStream stream, const int32_t* sorted, const uint16_t* ew,
                uint32_t rows, uint32_t hidden, uint32_t top_k) {
    const uint32_t slots = rows * top_k;
    hipLaunchKernelGGL(mailbox_add_wait_kernel, dim3(1), dim3(1), 0, (hipStream_t)stream,
                       static_cast<uint8_t*>(d->mailbox), d->sequence, d->published, d->rank, d->helpers);
    if (hipMemsetAsync(d->rmask, 0, (size_t)slots * sizeof(uint32_t), (hipStream_t)stream) != hipSuccess ||
        hipMemsetAsync(d->inv, 0xff, (size_t)slots * sizeof(int32_t), (hipStream_t)stream) != hipSuccess)
        return RAD_E_DEVICE;
    hipLaunchKernelGGL(mailbox_route_mark_kernel, dim3((slots + 255u) / 256u), dim3(256), 0,
                       (hipStream_t)stream, static_cast<uint8_t*>(d->mailbox),
                       d->sequence, d->published, d->rank, d->helpers, ew, d->rmask, rows, hidden, top_k);
    hipLaunchKernelGGL(mailbox_inverse_kernel, dim3((slots + 255u) / 256u), dim3(256), 0,
                       (hipStream_t)stream, sorted, d->inv, slots);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int add_launch(const RadArgs* a, RadStream stream) {
    if (!a || !a->instance || !stream || a->n_t != 6) return RAD_E_INVAL;
    auto* d = static_cast<R3MoeMailboxDevice*>(a->instance);
    const long long hidden = geti(a, "n");
    const long long top_k = geti(a, "top_k");
    const long long rows = a->t[5].rank > 0 ? a->t[5].shape[0] : 0;
    const bool has_shared = a->t[3].data || a->t[4].data;
    const bool ok = rows > 0 && rows <= (long long)d->max_rows &&
        hidden == (long long)kHidden && top_k == (long long)kTopK &&
        a->t[0].data && a->t[1].data && a->t[2].data && a->t[5].data &&
        a->t[0].dtype == RAD_I32 && a->t[1].dtype == RAD_BF16 &&
        a->t[2].dtype == RAD_BF16 && a->t[5].dtype == RAD_BF16 &&
        a->t[0].rank >= 1 && a->t[1].rank >= 2 && a->t[2].rank >= 2 &&
        a->t[5].rank >= 2 && a->t[0].shape[0] >= rows * top_k &&
        a->t[1].shape[0] >= rows * top_k && a->t[1].shape[1] == hidden &&
        a->t[2].shape[0] == rows && a->t[2].shape[1] == top_k &&
        a->t[5].shape[1] == hidden &&
        (!has_shared || (a->t[3].data && a->t[4].data &&
                         a->t[3].dtype == RAD_BF16 && a->t[4].dtype == RAD_BF16 &&
                         a->t[3].rank >= 2 && a->t[4].rank >= 2 &&
                         a->t[3].shape[0] == rows && a->t[3].shape[1] == hidden &&
                         a->t[4].shape[0] == rows && a->t[4].shape[1] == 1)) &&
        rad_tensor_is_contiguous(&a->t[0]) && rad_tensor_is_contiguous(&a->t[1]) &&
        rad_tensor_is_contiguous(&a->t[2]) && rad_tensor_is_contiguous(&a->t[5]) &&
        (!has_shared || (rad_tensor_is_contiguous(&a->t[3]) &&
                         rad_tensor_is_contiguous(&a->t[4]))) &&
        (!d->mask_only || d->mask_ref);
    if (!ok) return RAD_E_SHAPE;

    if (launch_maps(d, stream, static_cast<const int32_t*>(a->t[0].data),
                    static_cast<const uint16_t*>(a->t[2].data),
                    (uint32_t)rows, (uint32_t)hidden, (uint32_t)top_k) != RAD_OK)
        return RAD_E_DEVICE;
    hipLaunchKernelGGL(mailbox_add_rows_kernel, dim3((uint32_t)rows), dim3(256), 0,
                       (hipStream_t)stream, static_cast<uint8_t*>(d->mailbox),
                       d->published, d->rank, d->helpers, d->inv, d->rmask,
                       static_cast<const uint16_t*>(a->t[1].data),
                       static_cast<const uint16_t*>(a->t[2].data),
                       static_cast<const uint16_t*>(a->t[3].data),
                       static_cast<const uint16_t*>(a->t[4].data),
                       static_cast<uint16_t*>(a->t[5].data),
                       d->mask_ref, d->mask_only,
                       (uint32_t)rows, (uint32_t)hidden, (uint32_t)top_k);
    hipLaunchKernelGGL(mailbox_add_finish_kernel, dim3(1), dim3(kMaxRanks), 0, (hipStream_t)stream,
                       d->published);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int shadow_launch(const RadArgs* a, RadStream stream) {
    if (!a || !a->instance || !stream || a->n_t != 3) return RAD_E_INVAL;
    auto* d = static_cast<R3MoeMailboxDevice*>(a->instance);
    if (!d->shadow || !d->shadow_stats || !d->shadow_first) return RAD_E_STATE;
    const long long hidden = geti(a, "n");
    const long long top_k = geti(a, "top_k");
    if (hidden != (long long)kHidden || top_k != (long long)kTopK ||
        !a->t[0].data || !a->t[1].data || a->t[0].dtype != RAD_I32 ||
        a->t[1].dtype != RAD_BF16 || a->t[1].rank < 2 ||
        a->t[1].shape[1] != hidden || !rad_tensor_is_contiguous(&a->t[0]) ||
        !rad_tensor_is_contiguous(&a->t[1]) || !a->t[2].data || a->t[2].dtype != RAD_BF16 ||
        !rad_tensor_is_contiguous(&a->t[2]))
        return RAD_E_SHAPE;
    const long long slots = a->t[1].shape[0];
    if (slots <= 0 || slots % top_k != 0 || slots / top_k > (long long)d->max_rows)
        return RAD_E_SHAPE;
    const uint32_t rows = (uint32_t)(slots / top_k);

    if (launch_maps(d, stream, static_cast<const int32_t*>(a->t[0].data),
                    static_cast<const uint16_t*>(a->t[2].data), rows, (uint32_t)hidden,
                    (uint32_t)top_k) != RAD_OK)
        return RAD_E_DEVICE;
    hipLaunchKernelGGL(mailbox_shadow_rows_kernel, dim3(rows), dim3(256), 0,
                       (hipStream_t)stream, static_cast<uint8_t*>(d->mailbox),
                       d->sequence, d->published, d->rank, d->helpers, d->inv, d->rmask,
                       static_cast<const uint16_t*>(a->t[1].data),
                       static_cast<const uint16_t*>(a->t[2].data),
                       rows, (uint32_t)hidden, (uint32_t)top_k,
                       d->shadow_stats, d->shadow_first);
    hipLaunchKernelGGL(mailbox_add_finish_kernel, dim3(1), dim3(kMaxRanks), 0, (hipStream_t)stream,
                       d->published);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

const RadKernelInfo kPublish = [] {
    RadKernelInfo x{};
    x.name = "r3_moe_mailbox_publish";
    x.op = "moe_mailbox_publish";
    x.family = "moe";
    x.computes = "GPU-only remote MoE mailbox publication";
    x.shape = "decode rows, hidden=2560, top_k=10";
    x.dtypes = "i32,bf16,bf16,f8e4m3,f32,f8e4m3";
    x.domain = RAD_DOMAIN_DEVICE;
    x.priority = 1000;
    x.init = mailbox_init;
    x.launch = publish_launch;
    return x;
}();

const RadKernelInfo kAdd = [] {
    RadKernelInfo x{};
    x.name = "r3_moe_mailbox_add";
    x.op = "moe_mailbox_add";
    x.family = "moe";
    x.computes = "GPU-only exact slot-order merge of local and remote MoE rows";
    x.shape = "decode rows, hidden=2560, top_k=10";
    x.dtypes = "bf16";
    x.domain = RAD_DOMAIN_DEVICE;
    x.priority = 1000;
    x.init = mailbox_init;
    x.launch = add_launch;
    return x;
}();

const RadKernelInfo kMaskRef = [] {
    RadKernelInfo x{};
    x.name = "r3_moe_mailbox_mask_ref";
    x.op = "moe_mailbox_mask_ref";
    x.family = "moe";
    x.computes = "diagnostic engine-row capture/restore around masked rescatter";
    x.shape = "decode rows, hidden=2560, top_k=10";
    x.dtypes = "i32,bf16";
    x.domain = RAD_DOMAIN_DEVICE;
    x.priority = 1000;
    x.init = mailbox_init;
    x.launch = mask_ref_launch;
    return x;
}();

const RadKernelInfo kShadow = [] {
    RadKernelInfo x{};
    x.name = "r3_moe_mailbox_shadow";
    x.op = "moe_mailbox_shadow";
    x.family = "moe";
    x.computes = "diagnostic compare of helper and engine unweighted down rows";
    x.shape = "decode rows, hidden=2560, top_k=10";
    x.dtypes = "i32,bf16";
    x.domain = RAD_DOMAIN_DEVICE;
    x.priority = 1000;
    x.init = mailbox_init;
    x.launch = shadow_launch;
    return x;
}();

}  // namespace

extern "C" int r3_moe_mailbox_schema_count(void) { return NELEM(kSchemas); }
extern "C" const RadOpSchema* r3_moe_mailbox_schema_at(int i) {
    return (i < 0 || i >= NELEM(kSchemas)) ? nullptr : &kSchemas[i];
}
extern "C" const RadKernelInfo* r3_moe_mailbox_row(int i) {
    if (i == 0) return &kPublish;
    if (i == 1) return &kAdd;
    if (i == 2) return &kShadow;
    if (i == 3) return &kMaskRef;
    return nullptr;
}
