#include <cstdio>
#include "helper/moe_remote_hook.h"
#include "helper/moe_remote_shm.h"
#include "rad_abi.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>

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
    {"out", RAD_OPD_INOUT, 0},
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
     "wait for the helper GPU lane and add weighted remote expert rows after the engine gather"},
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
    if (rows == 0 || rows > max_rows || rows > kMaxRows) return;

    Route* route = reinterpret_cast<Route*>(base + h->route_offset[rank]);
    if (threadIdx.x == 0) {
        const uint32_t nroute = rows * top_k;
        for (uint32_t i = 0; i < nroute; ++i) {
            const int32_t le = ids[i] - expert_base;
            if (le < 0 || (uint32_t)le >= n_expert || (uint32_t)le >= kExperts ||
                !owner_has(owner, layer, (uint32_t)le))
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

__device__ __forceinline__ float bf16_product(float a, float b) {
    return bf16_to_f32(f32_to_bf16_rn(a * b));
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

__global__ void mailbox_add_rows_kernel(
    uint8_t* base, const uint32_t* sequence, const uint32_t* published, uint32_t rank,
    uint16_t* out, uint32_t rows, uint32_t hidden, uint32_t top_k) {
    if (!*published) return;
    Header* h = reinterpret_cast<Header*>(base);
    const uint32_t seq = *sequence;
    if ((int32_t)(load_sys_acq(&h->lane[rank].done_sequence) - seq) < 0) __builtin_trap();

    Request* ring = reinterpret_cast<Request*>(base + h->ring_offset);
    const Request* q = ring + rank * kLaneSlots + ((seq - 1u) % kLaneSlots);
    const uint32_t nr = q->route_count;
    const uint32_t token = blockIdx.x;
    if (q->sequence != seq || q->rank != rank || q->rows != rows || q->hidden != hidden ||
        q->status != 0 || nr == 0 || nr > rows * top_k || token >= rows)
        return;

    const Route* route = reinterpret_cast<const Route*>(base + q->route_offset);
    __shared__ uint32_t route_index[kTopK];
    if (threadIdx.x == 0) {
        for (uint32_t slot = 0; slot < top_k; ++slot) route_index[slot] = 0xffffffffu;
        for (uint32_t j = 0; j < nr; ++j) {
            const Route r = route[j];
            if (r.token < 0 || (uint32_t)r.token >= rows || r.slot >= top_k) __builtin_trap();
            if ((uint32_t)r.token != token) continue;
            if (route_index[r.slot] != 0xffffffffu) __builtin_trap();
            route_index[r.slot] = j;
        }
    }
    __syncthreads();

    uint16_t* dst = out + (uint64_t)token * hidden;
    const uint16_t* result = reinterpret_cast<const uint16_t*>(
        base + h->result_offset[rank * kMaxRanks + rank]);
    for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        float acc = bf16_to_f32(dst[col]);
        bool any = false;
        for (uint32_t slot = 0; slot < top_k; ++slot) {
            const uint32_t j = route_index[slot];
            if (j == 0xffffffffu) continue;
            any = true;
            const Route r = route[j];
            const uint32_t flat = token * top_k + slot;
            const float weight = bf16_to_f32(r.weight_bf16);
            const float value = bf16_to_f32(result[(uint64_t)flat * hidden + col]);
            acc += bf16_product(weight, value);
        }
        if (any) dst[col] = f32_to_bf16_rn(acc);
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
            shadow_claim(first, q, r, __int_as_float(0x7f800000), __int_as_float(0x7f800000),
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
            shadow_claim(first, q, r, rel, sm[0], local, helper);
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

int add_launch(const RadArgs* a, RadStream stream) {
    if (!a || !a->instance || !stream || a->n_t != 1) return RAD_E_INVAL;
    auto* d = static_cast<R3MoeMailboxDevice*>(a->instance);
    const long long hidden = geti(a, "n");
    const long long top_k = geti(a, "top_k");
    const long long rows = a->t[0].rank > 0 ? a->t[0].shape[0] : 0;
    auto numel = [](const RadTensor& t) {
        long long n = 1;
        for (int i = 0; i < t.rank; ++i) n *= t.shape[i];
        return n;
    };
    const bool ok = rows > 0 && rows <= (long long)d->max_rows &&
        hidden == (long long)kHidden && top_k == (long long)kTopK &&
        a->t[0].data && a->t[0].rank >= 2 && numel(a->t[0]) >= rows * hidden &&
        a->t[0].shape[a->t[0].rank - 1] == hidden && a->t[0].dtype == RAD_BF16 &&
        rad_tensor_is_contiguous(&a->t[0]);
    if (!ok) {
        std::fprintf(stderr, "libr3: moe_mailbox_add refused rows=%lld max_rows=%u hidden=%lld top_k=%lld "
                     "out rank %d numel %lld last %lld dtype %u\n",
                     rows, (unsigned)d->max_rows, hidden, top_k,
                     a->t[0].rank, numel(a->t[0]),
                     a->t[0].rank ? (long long)a->t[0].shape[a->t[0].rank - 1] : -1LL,
                     (unsigned)a->t[0].dtype);
        return RAD_E_SHAPE;
    }

    hipLaunchKernelGGL(mailbox_add_wait_kernel, dim3(1), dim3(1), 0, (hipStream_t)stream,
                       static_cast<uint8_t*>(d->mailbox), d->sequence, d->published, d->rank);
    hipLaunchKernelGGL(mailbox_add_rows_kernel, dim3((uint32_t)rows), dim3(256), 0,
                       (hipStream_t)stream, static_cast<uint8_t*>(d->mailbox),
                       d->sequence, d->published, d->rank,
                       static_cast<uint16_t*>(a->t[0].data),
                       (uint32_t)rows, (uint32_t)hidden, (uint32_t)top_k);
    hipLaunchKernelGGL(mailbox_add_finish_kernel, dim3(1), dim3(1), 0, (hipStream_t)stream,
                       d->published);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
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
    x.dtypes = "i32,bf16,bf16";
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
    x.computes = "GPU-only weighted remote MoE add after engine gather";
    x.shape = "decode rows, hidden=2560, top_k=10";
    x.dtypes = "bf16";
    x.domain = RAD_DOMAIN_DEVICE;
    x.priority = 1000;
    x.init = mailbox_init;
    x.launch = add_launch;
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
    return nullptr;
}
