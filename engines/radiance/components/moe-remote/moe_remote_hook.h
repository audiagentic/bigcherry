#pragma once

#include <cstdint>
#include <cstddef>

struct R3MoeShadowLayerStats {
    uint64_t rows = 0;
    uint64_t elems = 0;
    float sum_abs = 0.0f;
    float diff2 = 0.0f;
    float ref2 = 0.0f;
    uint32_t max_abs_bits = 0;
    uint32_t bad_rows = 0;
};

struct R3MoeShadowFirst {
    uint32_t claimed = 0;
    uint32_t layer = 0;
    uint32_t rank = 0;
    uint32_t row_base = 0;
    int32_t token = -1;
    uint16_t slot = 0;
    uint16_t reserved0 = 0;
    int32_t expert = -1;
    float rel_l2 = 0.0f;
    float max_abs = 0.0f;
    float local[4]{};
    float helper[4]{};
};

/* Rank-local, GPU-resident cell. The queue-ordered setter overwrites it only
 * after the previous pass's lane-1-to-lane-0 join. Mailbox requests capture
 * epoch/slot before they are released to the helpers. */
struct StepTicket {
    uint32_t placement_epoch = 0;
    uint8_t generation_slot = 0;
    uint8_t reserved[3]{};
    uint64_t ticket_id = 0;
};
static_assert(sizeof(StepTicket) == 16 && offsetof(StepTicket, ticket_id) == 8);

/* One completion slot per outstanding parent per rank, indexed by
 * ticket_id % capacity. capacity is the admission ledger's ceiling
 * (Ledger::capacity()), which rad0002 derives from the engine as its
 * RouteSlot ring extent plus its issue-ahead depth (ADMISSION.md); live
 * ticket IDs are consecutive and never exceed it, so slots never alias.
 * Two rings of this layout per rank:
 *  - device memory: the ticket setter arms the slot on a parent's first pass;
 *    publish and finish count with device-scope atomics, no host traffic;
 *  - host-mapped: once per parent step the seal kernel copies the device slot
 *    there (fields, then ticket_id with a system-scope release) ahead of the
 *    parent's terminal event; only the host ledger reads it. */
struct ConsumptionStamp {
    uint64_t ticket_id = 0;
    uint32_t placement_epoch = 0;
    uint32_t generation_slot = 0;
    uint32_t published_count = 0;
    uint32_t consumed_count = 0;
};
static_assert(sizeof(ConsumptionStamp) == 24);
/* Bound to each device generation after its bitmap copy on the private stream. */
struct R3MoeOwnerBinding {
    uint64_t epoch = 0;
    uint64_t digest = 0;
};
static_assert(sizeof(R3MoeOwnerBinding) == 16);
#if defined(__HIPCC__)
#define R3_MOE_OWNER_HD __host__ __device__
#else
#define R3_MOE_OWNER_HD
#endif
R3_MOE_OWNER_HD inline bool r3_moe_owner_binding_valid(const StepTicket& t,
                                                        const R3MoeOwnerBinding& b) {
    return t.ticket_id && t.placement_epoch && t.generation_slot < 2 &&
           !t.reserved[0] && !t.reserved[1] && !t.reserved[2] &&
           b.epoch == t.placement_epoch && b.digest != 0;
}
R3_MOE_OWNER_HD inline const uint8_t* r3_moe_select_owner_bitmap(
        const StepTicket& ticket, const R3MoeOwnerBinding* binding,
        const uint8_t* slot0, const uint8_t* slot1) {
    if (ticket.generation_slot > 1 || !binding) return nullptr;
    if (!r3_moe_owner_binding_valid(ticket, binding[ticket.generation_slot]))
        return nullptr;
    return ticket.generation_slot ? slot1 : slot0;
}
/* One host/device bit predicate; producer and fake tests share the exact
 * layer-major, helper-strided, byte-and-bit interpretation. */
R3_MOE_OWNER_HD inline bool r3_moe_owner_has(const uint8_t* bits,
        uint32_t layer,uint32_t expert,uint32_t experts) {
    if (!bits) return false;
    const uint64_t bit=(uint64_t)layer*experts+expert;
    return (bits[bit>>3] & uint8_t(1u<<(bit&7u))) != 0;
}
#undef R3_MOE_OWNER_HD
namespace r3 { namespace moe_shm { struct ConsumedRecords; } }
struct R3MoeMailboxDevice {
    void* mailbox = nullptr;
    const uint8_t* owner[2]{};  /* fixed pointers, indexed by this pass's ticket */
    const R3MoeOwnerBinding* owner_binding = nullptr; /* device-resident metadata per slot */
    StepTicket* ticket = nullptr;  /* fixed device address, independent per engine rank */
    ConsumptionStamp* consumed = nullptr;  /* mapped coherent rank-local completion ring (seal kernel writes) */
    ConsumptionStamp* stamps = nullptr;    /* device-memory counting ring, same index */
    r3::moe_shm::ConsumedRecords* consumed_rec = nullptr;  /* device memory: last consumed request a lane, last combine */
    uint32_t consumed_capacity = 0;
    uint32_t* sequence = nullptr;   /* [kLanes] request counters; this rank uses only its own slots' lanes */
    uint32_t* published = nullptr;  /* [kMaxSlots slots]: 0, or 1 + the lane class of this step's request; then [kMaxSlots]:
                                       the step's R3_MOE_WORKER_COMBINE generation, 0 when the step merges two rows; then
                                       [kCombineChunks]: the generation whose chunk c the engine has seen combined (RAD42, device
                                       mirror of Header::chunks.done, so only one block a chunk polls host memory) */
    uint16_t* mask_ref = nullptr;
    int32_t* inv = nullptr;     /* [max_rows * top_k] engine sorted row of each locally computed slot, else -1. */
    uint32_t* rmask = nullptr;  /* [max_rows * top_k] 1 where the helper computed the slot. */
    R3MoeShadowLayerStats* shadow_stats = nullptr;
    R3MoeShadowFirst* shadow_first = nullptr;
    uint32_t max_rows = 0;
    uint32_t hidden = 0;    /* the segment's model shape: every launch is checked against it */
    uint32_t top_k = 0;
    uint32_t experts = 0;
    uint32_t rank = 0;
    uint32_t world = 0;   /* engine TP ranks; the helper always computes full expert width, and a layer's consumer
                             rank (consumer_of(layer, world)) is the one that publishes and adds it (RAD49) */
    uint32_t helpers = 1; /* helper h is served from this rank's slot slot_of(h, rank, world) */
    uint32_t* heat = nullptr;   /* [kMaxLayers][kExperts] route counts (R3_MOE_WORKER_HISTOGRAM) */
    uint32_t shadow = 0;
    uint32_t mask_only = 0;
    uint32_t cpu_ref = 0;
    uint32_t combine = 0;   /* R3_MOE_WORKER_COMBINE=1: ask helper 0 for one combined fp16 row a token on the large lane */
};

/* Called by the engine hook before every run_pass. An ordinary (non-chained,
 * barrier-bit) dispatch is used so queued tape plays cannot overtake it. */
int r3_moe_ticket_launch(const R3MoeMailboxDevice* d, StepTicket value, void* stream);
/* After the parent's last GPU effect on this rank, before its terminal event:
 * one single-thread kernel copies the parent's device stamp and the rank's
 * consumed lane/combine records to host-mapped memory (posted stores, one
 * system-scope release). The terminal event's system-scope release then makes
 * them visible to the host before the event reads complete. */
int r3_moe_seal_launch(const R3MoeMailboxDevice* d, StepTicket value, void* stream);

/* Controller-facing quiescence API. Absolute CLOCK_MONOTONIC nanosecond
 * deadline; only these APIs wait. No GPU device synchronization is used. */
struct R3MoeDrainSnapshot {
    uint64_t closure_id;
    uint64_t cutoff_ticket_id;
    uint32_t old_epoch;
    uint32_t outstanding;
    uint32_t high_water;
    uint32_t capacity;
};
extern "C" int r3_moe_close_admission(R3MoeDrainSnapshot* snapshot);
extern "C" int r3_moe_wait_drained(const R3MoeDrainSnapshot* snapshot, uint64_t deadline_monotonic_ns);
/* Nonblocking, 1 only when every ticket admitted under this exact closed
 * snapshot has retired after every rank's seal/event. 0 = pending,
 * negative = invalid snapshot or fault; no helper drain or GPU sync. */
extern "C" int r3_moe_all_ranks_consumed(const R3MoeDrainSnapshot* snapshot);
extern "C" int r3_moe_reopen_admission(uint32_t epoch, uint32_t generation_slot);

extern "C" int r3_moe_mailbox_enabled(void);
extern "C" int r3_moe_mailbox_device(int rank, int world_size, R3MoeMailboxDevice** out);
extern "C" int r3_moe_mailbox_schema_count(void);

/* Transition-only API: asynchronous upload on a rank-private stream; event
 * query does not synchronize serving. Live coordinator wiring is later work. */
extern "C" int r3_moe_owner_generation_upload(int rank, uint32_t slot, uint64_t epoch, uint64_t digest);
extern "C" int r3_moe_owner_generation_ready(int rank, uint32_t slot, uint64_t epoch, uint64_t digest);
