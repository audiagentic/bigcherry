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
};
static const RadParamSpec kJoinParams[] = {
    {"M", RAD_P_INT, RAD_REQUIRED},
    {"n", RAD_P_INT, RAD_REQUIRED},
};
static const RadOperandSpec kJoinOperands[] = {
    {"remote", RAD_OPD_OUT, 0},
};
static const RadOpSchema kSchemas[] = {
    {"moe_mailbox_publish", kPublishParams, NELEM(kPublishParams),
     kPublishOperands, NELEM(kPublishOperands),
     "publish helper-owned MoE routes and activations to the mapped GPU mailbox"},
    {"moe_mailbox_join", kJoinParams, NELEM(kJoinParams),
     kJoinOperands, NELEM(kJoinOperands),
     "wait for the helper GPU lane and copy its BF16 partial from the mapped mailbox"},
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
    uint8_t* base, const uint8_t* owner, uint32_t* sequence, uint32_t rank,
    int32_t* ids, const uint16_t* ew, const uint16_t* x,
    uint32_t layer, uint32_t rows, uint32_t hidden, uint32_t top_k,
    int32_t expert_base, uint32_t n_expert) {
    __shared__ uint32_t seq;
    __shared__ uint32_t nr;
    __shared__ uint8_t selected[kMaxRows];

    Header* h = reinterpret_cast<Header*>(base);
    if (threadIdx.x == 0) {
        seq = atomicAdd(sequence, 1u) + 1u;
        nr = 0;
        if (seq == 0) __builtin_trap();
    }
    for (uint32_t r = threadIdx.x; r < rows; r += blockDim.x) selected[r] = 0;
    __syncthreads();

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
            route[p].reserved = 0;
            selected[i / top_k] = 1;
            ids[i] = -1;
        }
    }
    __syncthreads();

    uint16_t* activation = reinterpret_cast<uint16_t*>(base + h->activation_offset[rank]);
    const uint64_t total = (uint64_t)rows * hidden;
    for (uint64_t i = threadIdx.x; i < total; i += blockDim.x) {
        const uint32_t row = (uint32_t)(i / hidden);
        if (selected[row]) activation[i] = x[i];
    }

    Request* ring = reinterpret_cast<Request*>(base + h->ring_offset);
    Request* q = ring + rank * kLaneSlots + ((seq - 1u) % kLaneSlots);
    if (threadIdx.x == 0) {
        q->completion_sequence = 0;
        q->status = 0;
        q->layer = layer;
        q->rank = rank;
        q->rank_count = kMaxRanks;
        q->rows = rows;
        q->hidden = hidden;
        q->route_count = nr;
        q->reserved0 = 0;
        q->route_offset = h->route_offset[rank];
        q->activation_offset = h->activation_offset[rank];
        q->activation_stride = hidden * (uint32_t)sizeof(uint16_t);
        for (uint32_t r = 0; r < kMaxRanks; ++r) {
            q->result_offset[r] = h->result_offset[rank * kMaxRanks + r];
            q->result_stride[r] = hidden * (uint32_t)sizeof(uint16_t);
        }
        q->sequence = seq;
    }

    /* Every lane store must reach the host-mapped page before the release publication. This is
     * the same all-thread drain + one-thread system-scope flag pattern used by libr4d all-reduce. */
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
        store_sys_rel(&h->lane[rank].request_sequence, seq);
}

__global__ void mailbox_join_kernel(
    uint8_t* base, const uint32_t* sequence, uint32_t rank,
    uint16_t* out, uint64_t elems) {
    __shared__ uint32_t seq;
    Header* h = reinterpret_cast<Header*>(base);
    if (threadIdx.x == 0) {
        seq = *sequence;
        unsigned long long spins = 0;
        while ((int32_t)(load_sys_acq(&h->lane[rank].done_sequence) - seq) < 0) {
            __builtin_amdgcn_s_sleep(2);
            if (++spins > 4000000000ULL) __builtin_trap();
        }
    }
    __syncthreads();

    const uint16_t* src = reinterpret_cast<const uint16_t*>(
        base + h->result_offset[rank * kMaxRanks + rank]);
    for (uint64_t i = threadIdx.x; i < elems; i += blockDim.x) out[i] = src[i];
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
    if (!a || !a->instance || !stream || a->n_t != 3) return RAD_E_INVAL;
    auto* d = static_cast<R3MoeMailboxDevice*>(a->instance);
    const long long layer = geti(a, "layer");
    const long long rows = geti(a, "M");
    const long long hidden = geti(a, "n");
    const long long top_k = geti(a, "top_k");
    const long long expert_base = geti(a, "expert_base");
    const long long n_expert = geti(a, "n_expert");
    if (layer < 0 || layer >= (long long)kMaxLayers || rows <= 0 || rows > (long long)kMaxRows ||
        hidden != (long long)kHidden || top_k != (long long)kTopK ||
        expert_base < 0 || n_expert <= 0 || n_expert > (long long)kExperts ||
        !a->t[0].data || !a->t[1].data || !a->t[2].data ||
        a->t[0].dtype != RAD_I32 || a->t[1].dtype != RAD_BF16 || a->t[2].dtype != RAD_BF16 ||
        !rad_tensor_is_contiguous(&a->t[0]) || !rad_tensor_is_contiguous(&a->t[1]) ||
        !rad_tensor_is_contiguous(&a->t[2]))
        return RAD_E_SHAPE;

    hipLaunchKernelGGL(mailbox_publish_kernel, dim3(1), dim3(256), 0, (hipStream_t)stream,
                       static_cast<uint8_t*>(d->mailbox), d->owner, d->sequence, d->rank,
                       static_cast<int32_t*>(a->t[0].data),
                       static_cast<const uint16_t*>(a->t[1].data),
                       static_cast<const uint16_t*>(a->t[2].data),
                       (uint32_t)layer, (uint32_t)rows, (uint32_t)hidden, (uint32_t)top_k,
                       (int32_t)expert_base, (uint32_t)n_expert);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int join_launch(const RadArgs* a, RadStream stream) {
    if (!a || !a->instance || !stream || a->n_t != 1) return RAD_E_INVAL;
    auto* d = static_cast<R3MoeMailboxDevice*>(a->instance);
    const long long rows = geti(a, "M");
    const long long hidden = geti(a, "n");
    if (rows <= 0 || rows > (long long)kMaxRows || hidden != (long long)kHidden ||
        !a->t[0].data || a->t[0].dtype != RAD_BF16 || !rad_tensor_is_contiguous(&a->t[0]))
        return RAD_E_SHAPE;
    hipLaunchKernelGGL(mailbox_join_kernel, dim3(1), dim3(256), 0, (hipStream_t)stream,
                       static_cast<uint8_t*>(d->mailbox), d->sequence, d->rank,
                       static_cast<uint16_t*>(a->t[0].data), (uint64_t)rows * (uint64_t)hidden);
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

const RadKernelInfo kJoin = [] {
    RadKernelInfo x{};
    x.name = "r3_moe_mailbox_join";
    x.op = "moe_mailbox_join";
    x.family = "moe";
    x.computes = "GPU-only remote MoE mailbox completion join";
    x.shape = "decode rows, hidden=2560";
    x.dtypes = "bf16";
    x.domain = RAD_DOMAIN_DEVICE;
    x.priority = 1000;
    x.init = mailbox_init;
    x.launch = join_launch;
    return x;
}();

}  // namespace

extern "C" int r3_moe_mailbox_schema_count(void) { return NELEM(kSchemas); }
extern "C" const RadOpSchema* r3_moe_mailbox_schema_at(int i) {
    return (i < 0 || i >= NELEM(kSchemas)) ? nullptr : &kSchemas[i];
}
extern "C" const RadKernelInfo* r3_moe_mailbox_row(int i) {
    if (i == 0) return &kPublish;
    if (i == 1) return &kJoin;
    return nullptr;
}
