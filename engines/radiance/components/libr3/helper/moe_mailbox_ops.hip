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
};
static const RadOpSchema kSchemas[] = {
    {"moe_mailbox_publish", kPublishParams, NELEM(kPublishParams),
     kPublishOperands, NELEM(kPublishOperands),
     "publish helper-owned MoE routes and activations to the mapped GPU mailbox"},
    {"moe_mailbox_add", kAddParams, NELEM(kAddParams),
     kAddOperands, NELEM(kAddOperands),
     "merge local and remote expert rows in original slot order with moe_gather arithmetic"},
    {"moe_mailbox_mask_ref", kMaskRefParams, NELEM(kMaskRefParams),
     kMaskRefOperands, NELEM(kMaskRefOperands),
     "capture or restore engine down rows for the mask-only diagnostic"},
    {"moe_mailbox_shadow", kShadowParams, NELEM(kShadowParams),
     kShadowOperands, NELEM(kShadowOperands),
     "compare helper unweighted down rows against the engine rows without changing model output"},
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

__global__ void mailbox_publish_kernel(
    uint8_t* base, const uint8_t* owner, uint32_t* sequence, uint32_t* published, uint32_t rank,
    int32_t* ids, const uint16_t* ew, const uint16_t* x,
    const uint8_t* aq, const float* as, const uint8_t* aq_base,
    uint32_t layer, uint32_t rows, uint32_t hidden, uint32_t top_k,
    int32_t expert_base, uint32_t n_expert, uint32_t max_rows,
    uint32_t shadow, uint32_t stage_bf16) {
    __shared__ uint32_t seq;
    __shared__ uint32_t nr;

    Header* h = reinterpret_cast<Header*>(base);
    if (threadIdx.x == 0) {
        *published = 0;
        nr = 0;
    }
    __syncthreads();
    if (rows == 0 || rows > max_rows || rows > kMaxRows ||
        layer >= h->layer_count || !load_sys_acq(&h->layer_map_ready[layer])) return;

    const int32_t* slot_map = reinterpret_cast<const int32_t*>(
        base + h->layer_map_offset) + (uint64_t)layer * kExperts;
    Route* route = reinterpret_cast<Route*>(base + h->route_offset[rank]);
    if (threadIdx.x == 0) {
        const uint32_t nroute = rows * top_k;
        for (uint32_t i = 0; i < nroute; ++i) {
            const int32_t le = ids[i] - expert_base;
            if (le < 0 || (uint32_t)le >= n_expert || (uint32_t)le >= kExperts ||
                slot_map[(uint32_t)le] < 0 || !owner_has(owner, layer, (uint32_t)le))
                continue;
            const uint32_t p = nr++;
            route[p].token = (int32_t)(i / top_k);
            route[p].expert = le;
            route[p].weight_bf16 = ew[i];
            route[p].slot = (uint16_t)(i % top_k);
            if (!shadow) ids[i] = -1;
        }
    }
    __syncthreads();
    if (nr == 0) return;

    if (threadIdx.x == 0) {
        seq = atomicAdd(sequence, 1u) + 1u;
        if (seq == 0) __builtin_trap();
    }
    __syncthreads();

    uint8_t* staged_q = reinterpret_cast<uint8_t*>(
        base + h->activation_offset[rank] + kActivationBf16Bytes);
    float* staged_s = reinterpret_cast<float*>(
        base + h->activation_offset[rank] + kActivationBf16Bytes + kActivationQBytes);
    const uint64_t q_total = (uint64_t)rows * hidden;
    for (uint64_t i = threadIdx.x; i < q_total; i += blockDim.x) staged_q[i] = aq[i];
    const uint32_t scale_cols = hidden / kActGroup;
    const uint64_t s_total = (uint64_t)rows * scale_cols;
    for (uint64_t i = threadIdx.x; i < s_total; i += blockDim.x) staged_s[i] = as[i];

    if (stage_bf16) {
        uint16_t* staged_x = reinterpret_cast<uint16_t*>(base + h->activation_offset[rank]);
        for (uint64_t i = threadIdx.x; i < q_total; i += blockDim.x) staged_x[i] = x[i];
    }

    Request* ring = reinterpret_cast<Request*>(base + h->ring_offset);
    Request* q = ring + rank * kLaneSlots + ((seq - 1u) % kLaneSlots);
    if (threadIdx.x == 0) {
        const uintptr_t qptr = reinterpret_cast<uintptr_t>(aq);
        const uintptr_t qbase = reinterpret_cast<uintptr_t>(aq_base);
        if (qptr < qbase || ((qptr - qbase) % hidden) != 0) __builtin_trap();
        q->completion_sequence = 0;
        q->status = 0;
        q->layer = layer;
        q->rank = rank;
        q->rank_count = kMaxRanks;
        q->rows = rows;
        q->hidden = hidden;
        q->route_count = nr;
        q->row_base = (uint32_t)((qptr - qbase) / hidden);
        q->route_offset = h->route_offset[rank];
        q->activation_offset = h->activation_offset[rank];
        q->activation_q_offset = h->activation_offset[rank] + kActivationBf16Bytes;
        q->activation_s_offset =
            h->activation_offset[rank] + kActivationBf16Bytes + kActivationQBytes;
        q->activation_stride = hidden * (uint32_t)sizeof(uint16_t);
        q->activation_q_stride = hidden;
        q->activation_s_stride = scale_cols * (uint32_t)sizeof(float);
        for (uint32_t r = 0; r < kMaxRanks; ++r) {
            q->result_offset[r] = h->result_offset[rank * kMaxRanks + r];
            q->result_stride[r] = hidden * (uint32_t)sizeof(uint16_t);
        }
        q->sequence = seq;
    }

    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        store_sys_rel(&h->lane[rank].request_sequence, seq);
        *published = 1;
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

__global__ void mailbox_add_wait_kernel(
    uint8_t* base, const uint32_t* sequence, const uint32_t* published, uint32_t rank) {
    if (blockIdx.x || threadIdx.x || !*published) return;
    Header* h = reinterpret_cast<Header*>(base);
    const uint32_t seq = *sequence;
    unsigned long long spins = 0;
    while ((int32_t)(load_sys_acq(&h->lane[rank].done_sequence) - seq) < 0) {
        __builtin_amdgcn_s_sleep(2);
        if (++spins > 4000000000ULL) __builtin_trap();
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

__global__ void mailbox_add_rows_kernel(
    uint8_t* base, const uint32_t* sequence, const uint32_t* published, uint32_t rank,
    const int32_t* sorted, const uint16_t* down, const uint16_t* ew,
    const uint16_t* shared, const uint16_t* shared_gate, uint16_t* out,
    const uint16_t* mask_ref, uint32_t mask_only, uint16_t* seen,
    uint32_t rows, uint32_t hidden, uint32_t top_k) {
    const uint32_t token = blockIdx.x;
    if (token >= rows) return;

    Header* h = reinterpret_cast<Header*>(base);
    __shared__ int32_t local_index[kTopK];
    __shared__ uint8_t remote_slot[kTopK];
    __shared__ uint32_t nr;
    __shared__ const Route* route;
    if (threadIdx.x == 0) {
        for (uint32_t slot = 0; slot < top_k; ++slot) {
            local_index[slot] = -1;
            remote_slot[slot] = 0;
        }
        nr = 0;
        route = nullptr;
        if (*published) {
            const uint32_t seq = *sequence;
            if ((int32_t)(load_sys_acq(&h->lane[rank].done_sequence) - seq) < 0) __builtin_trap();
            Request* ring = reinterpret_cast<Request*>(base + h->ring_offset);
            const Request* q = ring + rank * kLaneSlots + ((seq - 1u) % kLaneSlots);
            if (q->sequence != seq || q->rank != rank || q->rows != rows ||
                q->hidden != hidden || q->status != 0 ||
                q->route_count == 0 || q->route_count > rows * top_k)
                __builtin_trap();
            nr = q->route_count;
            route = reinterpret_cast<const Route*>(base + q->route_offset);
        }
    }
    __syncthreads();

    const int32_t lo = (int32_t)(token * top_k);
    const int32_t hi = lo + (int32_t)top_k;
    for (uint32_t i = threadIdx.x; i < rows * top_k; i += blockDim.x) {
        const int32_t flat = sorted[i];
        if (flat >= lo && flat < hi) local_index[(uint32_t)(flat - lo)] = (int32_t)i;
    }
    if (threadIdx.x == 0) {
        for (uint32_t j = 0; j < nr; ++j) {
            const Route r = route[j];
            if (r.token < 0 || (uint32_t)r.token >= rows || r.slot >= top_k) __builtin_trap();
            if ((uint32_t)r.token != token) continue;
            const uint32_t flat = token * top_k + r.slot;
            if (remote_slot[r.slot] || r.weight_bf16 != ew[flat]) __builtin_trap();
            remote_slot[r.slot] = 1;
        }
    }
    __syncthreads();
    if (mask_only && threadIdx.x == 0) {
        for (uint32_t slot = 0; slot < top_k; ++slot)
            if (remote_slot[slot] && local_index[slot] >= 0) __builtin_trap();
    }
    __syncthreads();

    const uint16_t* result = reinterpret_cast<const uint16_t*>(
        base + h->result_offset[rank * kMaxRanks + rank]);
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
                src = (mask_only == 1u ? mask_ref : result) + (uint64_t)flat * hidden;
            } else if (local_index[slot] >= 0) {
                src = down + (uint64_t)(uint32_t)local_index[slot] * hidden;
            }
            if (!src) continue;
            /* The helper's rows live in host-mapped memory the R9700 rewrites every request; a cached
             * read can return the previous request's row. Read them non-temporally (bypass caches). */
            const uint16_t v = (remote_slot[slot] && mask_only != 1u) ? __builtin_nontemporal_load(src + col) : src[col];
            if (seen && remote_slot[slot]) seen[(uint64_t)flat * hidden + col] = v;
            acc += bf16_to_f32(ew[flat]) * bf16_to_f32(v);
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

__device__ __forceinline__ void shadow_claim(
    R3MoeShadowFirst* first, const Request* q, const Route& r,
    float rel, float max_abs, const uint16_t* local, const uint16_t* helper) {
    if (atomicCAS(&first->claimed, 0u, 1u) != 0u) return;
    first->layer = q->layer;
    first->rank = q->rank;
    first->row_base = q->row_base;
    first->token = r.token;
    first->slot = r.slot;
    first->expert = r.expert;
    first->rel_l2 = rel;
    first->max_abs = max_abs;
    for (uint32_t i = 0; i < 4; ++i) {
        first->local[i] = bf16_to_f32(local[i]);
        first->helper[i] = bf16_to_f32(helper[i]);
    }
}

__global__ void mailbox_shadow_rows_kernel(
    uint8_t* base, const uint32_t* sequence, const uint32_t* published, uint32_t rank,
    const int32_t* sorted, const uint16_t* down, uint32_t rows, uint32_t hidden, uint32_t top_k,
    R3MoeShadowLayerStats* stats, R3MoeShadowFirst* first) {
    if (!*published) return;
    Header* h = reinterpret_cast<Header*>(base);
    const uint32_t seq = *sequence;
    if ((int32_t)(load_sys_acq(&h->lane[rank].done_sequence) - seq) < 0) __builtin_trap();

    Request* ring = reinterpret_cast<Request*>(base + h->ring_offset);
    const Request* q = ring + rank * kLaneSlots + ((seq - 1u) % kLaneSlots);
    const uint32_t j = blockIdx.x;
    if (q->sequence != seq || q->rank != rank || q->rows != rows || q->hidden != hidden ||
        q->status != 0 || j >= q->route_count || q->layer >= kMaxLayers)
        return;

    const Route* route = reinterpret_cast<const Route*>(base + q->route_offset);
    const Route r = route[j];
    if (r.token < 0 || (uint32_t)r.token >= rows || r.slot >= top_k) __builtin_trap();
    const int32_t flat = r.token * (int32_t)top_k + (int32_t)r.slot;

    __shared__ int32_t sorted_row;
    if (threadIdx.x == 0) {
        sorted_row = -1;
        for (uint32_t i = 0; i < rows * top_k; ++i) {
            if (sorted[i] == flat) {
                sorted_row = (int32_t)i;
                break;
            }
        }
    }
    __syncthreads();

    R3MoeShadowLayerStats* st = stats + q->layer;
    if (sorted_row < 0) {
        if (threadIdx.x == 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&st->rows), 1ull);
            atomicAdd(&st->bad_rows, 1u);
            const uint16_t zero[4] = {0, 0, 0, 0};
            shadow_claim(first + q->layer, q, r, __int_as_float(0x7f800000), __int_as_float(0x7f800000),
                         zero, zero);
        }
        return;
    }

    const uint16_t* local = down + (uint64_t)(uint32_t)sorted_row * hidden;
    const uint16_t* helper = reinterpret_cast<const uint16_t*>(
        base + q->result_offset[rank]) + (uint64_t)(uint32_t)flat * hidden;

    __shared__ float sa[256], sd2[256], sr2[256], sm[256];
    float a = 0.0f, d2 = 0.0f, r2 = 0.0f, m = 0.0f;
    for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        const float lv = bf16_to_f32(local[col]);
        const float hv = bf16_to_f32(helper[col]);
        const float d = lv - hv;
        const float ad = fabsf(d);
        a += ad;
        d2 += d * d;
        r2 += lv * lv;
        m = fmaxf(m, ad);
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
            shadow_claim(first + q->layer, q, r, rel, sm[0], local, helper);
        }
    }
}

__global__ void mailbox_add_finish_kernel(uint32_t* published) {
    if (!blockIdx.x && !threadIdx.x) *published = 0;
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

    hipLaunchKernelGGL(mailbox_publish_kernel, dim3(1), dim3(256), 0, (hipStream_t)stream,
                       static_cast<uint8_t*>(d->mailbox), d->owner, d->sequence, d->published, d->rank,
                       static_cast<int32_t*>(a->t[0].data),
                       static_cast<const uint16_t*>(a->t[1].data),
                       static_cast<const uint16_t*>(a->t[2].data),
                       static_cast<const uint8_t*>(a->t[3].data),
                       static_cast<const float*>(a->t[4].data),
                       static_cast<const uint8_t*>(a->t[5].data),
                       (uint32_t)layer, (uint32_t)rows, (uint32_t)hidden, (uint32_t)top_k,
                       (int32_t)expert_base, (uint32_t)n_expert, d->max_rows,
                       d->shadow, d->cpu_ref);
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

/* Diagnostic (R3_MOE_HELPER_DUMP=<dir>, mask-only modes only): once per rank, for the first request on layer
 * R3_MOE_HELPER_DUMP_LAYER (default 20), synchronise after the add and write the request's routes, the helper's
 * result rows and the engine's captured rows (mask_ref) to <dir>/r<rank>.{routes,helper,engine}.bin. Host side
 * only: device printf is not serviced on radiance's streams. */
void dump_rows(R3MoeMailboxDevice* d, RadStream stream, uint32_t rows, uint32_t hidden, uint32_t top_k) {
    static uint32_t done[kMaxRanks] = {};
    static bool layer_done[kMaxRanks][kMaxLayers] = {};
    const char* dir = std::getenv("R3_MOE_HELPER_DUMP");
    const char* le = std::getenv("R3_MOE_HELPER_DUMP_LAYER");
    const bool all = le && std::strcmp(le, "all") == 0;
    if (!dir || !*dir || !d->mask_only || !d->mask_ref || d->rank >= kMaxRanks || (!all && done[d->rank] >= 8u)) return;
    const uint32_t want = le && *le && !all ? (uint32_t)std::atoi(le) : 20u;
    if (hipStreamSynchronize((hipStream_t)stream) != hipSuccess) return;
    uint32_t seq = 0;
    if (hipMemcpy(&seq, d->sequence, sizeof(seq), hipMemcpyDeviceToHost) != hipSuccess || seq == 0) return;
    const char* sk = std::getenv("R3_MOE_HELPER_DUMP_SKIP");
    if (sk && *sk && seq < (uint32_t)std::atoi(sk)) return;
    auto* base = static_cast<uint8_t*>(d->mailbox);
    Header h{};
    Request q{};
    if (hipMemcpy(&h, base, sizeof(h), hipMemcpyDeviceToHost) != hipSuccess) return;
    const uint64_t qoff = h.ring_offset +
        (uint64_t)(d->rank * kLaneSlots + ((seq - 1u) % kLaneSlots)) * sizeof(Request);
    if (hipMemcpy(&q, base + qoff, sizeof(q), hipMemcpyDeviceToHost) != hipSuccess) return;
    if (q.sequence != seq || q.layer >= kMaxLayers || (all ? layer_done[d->rank][q.layer] : q.layer != want) || q.route_count == 0 || q.rows != rows) return;
    const size_t elems = (size_t)rows * top_k * hidden;
    std::vector<Route> routes(q.route_count);
    std::vector<uint16_t> helper(elems), engine(elems), seen(elems);
    if (hipMemcpy(routes.data(), base + q.route_offset, routes.size() * sizeof(Route),
                  hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(helper.data(), base + h.result_offset[d->rank * kMaxRanks + d->rank],
                  elems * sizeof(uint16_t), hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(engine.data(), d->mask_ref, elems * sizeof(uint16_t), hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(seen.data(), d->seen, elems * sizeof(uint16_t), hipMemcpyDeviceToHost) != hipSuccess)
        return;
    auto write = [&](const char* ext, const void* p, size_t n) {
        const std::string path = std::string(dir) + "/r" + std::to_string(d->rank) + "." + (all ? "L" + std::to_string(q.layer) : std::to_string(done[d->rank])) + "." + ext + ".bin";
        if (FILE* f = std::fopen(path.c_str(), "wb")) { std::fwrite(p, 1, n, f); std::fclose(f); }
    };
    write("routes", routes.data(), routes.size() * sizeof(Route));
    write("helper", helper.data(), helper.size() * sizeof(uint16_t));
    write("engine", engine.data(), engine.size() * sizeof(uint16_t));
    write("seen", seen.data(), seen.size() * sizeof(uint16_t));
    std::fprintf(stderr, "libr3: R3_MOE_HELPER dump rank %u layer %u seq %u rows %u routes %u hidden %u top_k %u\n",
                 d->rank, q.layer, seq, rows, q.route_count, hidden, top_k);
    ++done[d->rank];
    layer_done[d->rank][q.layer] = true;
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

    hipLaunchKernelGGL(mailbox_add_wait_kernel, dim3(1), dim3(1), 0, (hipStream_t)stream,
                       static_cast<uint8_t*>(d->mailbox), d->sequence, d->published, d->rank);
    hipLaunchKernelGGL(mailbox_add_rows_kernel, dim3((uint32_t)rows), dim3(256), 0,
                       (hipStream_t)stream, static_cast<uint8_t*>(d->mailbox),
                       d->sequence, d->published, d->rank,
                       static_cast<const int32_t*>(a->t[0].data),
                       static_cast<const uint16_t*>(a->t[1].data),
                       static_cast<const uint16_t*>(a->t[2].data),
                       static_cast<const uint16_t*>(a->t[3].data),
                       static_cast<const uint16_t*>(a->t[4].data),
                       static_cast<uint16_t*>(a->t[5].data),
                       d->mask_ref, d->mask_only, d->seen,
                       (uint32_t)rows, (uint32_t)hidden, (uint32_t)top_k);
    hipLaunchKernelGGL(mailbox_add_finish_kernel, dim3(1), dim3(1), 0, (hipStream_t)stream,
                       d->published);
    if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;
    dump_rows(d, stream, (uint32_t)rows, (uint32_t)hidden, (uint32_t)top_k);
    return RAD_OK;
}

int shadow_launch(const RadArgs* a, RadStream stream) {
    if (!a || !a->instance || !stream || a->n_t != 2) return RAD_E_INVAL;
    auto* d = static_cast<R3MoeMailboxDevice*>(a->instance);
    if (!d->shadow || !d->shadow_stats || !d->shadow_first) return RAD_E_STATE;
    const long long hidden = geti(a, "n");
    const long long top_k = geti(a, "top_k");
    if (hidden != (long long)kHidden || top_k != (long long)kTopK ||
        !a->t[0].data || !a->t[1].data || a->t[0].dtype != RAD_I32 ||
        a->t[1].dtype != RAD_BF16 || a->t[1].rank < 2 ||
        a->t[1].shape[1] != hidden || !rad_tensor_is_contiguous(&a->t[0]) ||
        !rad_tensor_is_contiguous(&a->t[1]))
        return RAD_E_SHAPE;
    const long long slots = a->t[1].shape[0];
    if (slots <= 0 || slots % top_k != 0 || slots / top_k > (long long)d->max_rows)
        return RAD_E_SHAPE;
    const uint32_t rows = (uint32_t)(slots / top_k);

    hipLaunchKernelGGL(mailbox_add_wait_kernel, dim3(1), dim3(1), 0, (hipStream_t)stream,
                       static_cast<uint8_t*>(d->mailbox), d->sequence, d->published, d->rank);
    hipLaunchKernelGGL(mailbox_shadow_rows_kernel, dim3((uint32_t)slots), dim3(256), 0,
                       (hipStream_t)stream, static_cast<uint8_t*>(d->mailbox),
                       d->sequence, d->published, d->rank,
                       static_cast<const int32_t*>(a->t[0].data),
                       static_cast<const uint16_t*>(a->t[1].data),
                       rows, (uint32_t)hidden, (uint32_t)top_k,
                       d->shadow_stats, d->shadow_first);
    hipLaunchKernelGGL(mailbox_add_finish_kernel, dim3(1), dim3(1), 0, (hipStream_t)stream,
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
