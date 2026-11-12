#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cstring>
#include <ctime>
#include <initializer_list>
#include <type_traits>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace r3::moe_shm {

constexpr uint64_t kMagic = 0x31454f4d33485253ull; /* "SRH3MOE1": reject unrelated/stale shm. */
constexpr uint32_t kVersion = 25;  /* RAD49: request slots (helper, consumer rank), full-width-once helper rows, per-rank combine; no v24 compatibility */
constexpr uint32_t kPage = 4096;
/* One existing helper segment-attach/startup window, shared by helper
 * helper publication/filter sealing and the engine's first rank bitmap upload. */
constexpr int kStartupWaitSeconds = 1800;
/* R3_MOE_WORKER_COMBINE: first element of a combined fp16 row the combiner could not hold in fp16 range (a NaN payload
 * no arithmetic produces; a real NaN converts to 0x7E00); the engine then reads that row from the helper planes. */
constexpr uint16_t kCombineFallback = 0x7E01;
/* Three axes (RAD23, RAD49): the engine's TP ranks, the helpers, and the request slots. A slot is one (helper, consumer
 * rank) pair: slot = helper * world + consumer, for every engine width, so one helper behind a TP2 engine uses slots 0
 * and 1 (one a rank) and two helpers behind a one-rank engine use slots 0 and 1 (one a helper), as before, and two helpers
 * behind TP2 use slots 0..3. Every slot has exactly one producer (its helper) and one consumer (its rank), so a lane's
 * sequence counter, its consumed record and its result plane each have one writer. A segment is laid out for the slots
 * it was made with (Header::slot_count = helpers * world). */
constexpr uint32_t kMaxEngineRanks = 2;  /* engine TP ranks: Header::rank_count, Request::world, RankState */
constexpr uint32_t kMaxHelpers = 4;      /* helper processes, helper_pid / helper_start, helper_count */
constexpr uint32_t kMaxSlots = 4;        /* request slots: helpers * engine ranks, at most */
static_assert(kMaxSlots >= kMaxEngineRanks && kMaxSlots >= kMaxHelpers, "a slot is needed for every engine rank and every helper");
constexpr uint32_t kResultPlanes = kMaxSlots * kMaxEngineRanks;  /* result planes at the bound: [slot][output rank] */
/* Request slots a segment needs: one per (helper, engine rank). */
constexpr uint32_t slots_for(uint32_t helpers, uint32_t world) { return helpers * world; }
/* Whether a helper count fits the slot bound behind an engine of this width. */
constexpr bool slots_fit(uint32_t helpers, uint32_t world) {
    return helpers >= 1 && helpers <= kMaxHelpers && world >= 1 && world <= kMaxEngineRanks && helpers * world <= kMaxSlots;
}
constexpr uint32_t slot_of(uint32_t helper, uint32_t consumer, uint32_t world) { return helper * world + consumer; }
/* A one-rank engine (world 1) is the common case: these then cost no division (they run in the helper lane kernels). */
constexpr uint32_t slot_helper(uint32_t slot, uint32_t world) { return world == 1u ? slot : slot / world; }
constexpr uint32_t slot_consumer(uint32_t slot, uint32_t world) { return world == 1u ? 0u : slot % world; }
/* FULL WIDTH ONCE (RAD49): the helper computes every routed expert it owns at the full intermediate width, and ONE engine
 * rank a layer stages the activations, publishes, waits and adds the helper rows; the other ranks mask the same routes out
 * of their local GEMMs and add nothing for them, so the TP all-reduce that follows sums every helper row exactly once.
 * The consumer alternates by layer, so the ranks' links share the helper traffic (two XTX: half each); every rank derives
 * it from the layer it is given and the world, so the ranks always agree. A one-rank engine is its own consumer. */
constexpr uint32_t consumer_of(uint32_t layer, uint32_t world) { return world > 1u ? layer % world : 0u; }
/* The role of one routed (token, top-k slot) on one engine rank for one layer (RAD49), the single definition the publish
 * kernel uses and the card-free exactly-once test (tp_exactly_once_test.cpp) checks: every rank counts the route's heat (so
 * rank 0's histogram covers every layer, consumed by it or not) and masks a helper-owned route out of its local GEMMs (unless
 * the shadow diagnostic keeps the local path); only the layer's consumer publishes it to the owning helper. */
struct RouteRole {
    bool count_heat, mask_local, publish;
};
/* `consumer`: this rank is the layer's consumer, consumer_of(layer, world) == rank, decided once a pass (the publish kernel's
 * route loop runs rows x top-k x helpers times on one block, so nothing per route may cost more than the route itself). */
constexpr RouteRole route_role(bool consumer, bool in_range, bool helper_owned, bool shadow) {
    return RouteRole{in_range, in_range && helper_owned && !shadow, in_range && helper_owned && consumer};
}
/* RAD49 fused gather (moe_mailbox_inject): on the layer's consumer, helper k's row for token t joins the engine's fused
 * MoE gather + reducing write as one more sorted row, at the first of t's slots whose route helper k computed
 * (rmask[t*K+s] == k+1, as route-mark writes it). Those slots were masked out of the local sort, so the slot is unclaimed,
 * and two helpers never share one. -1: helper k has no route of t. One definition for the inject kernel and the card-free
 * exactly-once test. */
constexpr int32_t inject_slot(const uint32_t* rmask_row, uint32_t top_k, uint32_t helper) {
    for (uint32_t s = 0; s < top_k; ++s)
        if (rmask_row[s] == helper + 1u) return (int32_t)s;
    return -1;
}
/* The router weight the injected slot takes: bf16 1.0, so the gather adds the helper's already weighted row unchanged. */
constexpr uint16_t kInjectWeightBf16 = 0x3F80;
/* The plane a slot writes for an output rank. Every reader and writer takes the index from here (the host layout, the device kernels
 * and the helper), so the convention lives in one place. A slot writes only its consumer's plane, [slot][slot_consumer]. */
constexpr uint32_t result_index(uint32_t slot, uint32_t out_rank) { return slot * kMaxEngineRanks + out_rank; }
/* R3_MOE_WORKER_COMBINE: the plane helper 0 writes the combined fp16 rows of a consumer rank into: the unused output-rank
 * plane of helper 0's slot for that consumer ([0][1] for a one-rank engine; [0][1] and [1][0] behind TP2). */
constexpr uint32_t combine_index(uint32_t consumer) { return result_index(consumer, (consumer + 1u) % kMaxEngineRanks); }
constexpr uint32_t kMaxLayers = 128;     /* protocol layer bound */
/* Lanes a producer rank (RAD12): up to kSmallClasses small lanes, each pre-queued by the helper at its own fixed row
 * count (Header::small_class_rows, ascending; a step goes to the first class that fits its rows), and one large lane
 * for the larger (prefill) steps, which the helper launches sized to the request when it arrives. A rank publishes one
 * request at a time, so every lane shares that rank's route/activation/result slots. Lane index = class * kMaxSlots + slot. */
constexpr uint32_t kSmallClasses = 4;
constexpr uint32_t kClasses = kSmallClasses + 1;          /* the last class is the large lane */
constexpr uint32_t kLanes = kMaxSlots * kClasses;
constexpr uint32_t kLaneSlots = 4;                        /* lane-private metadata ring; bulk slots stay per-rank. */
constexpr uint32_t kRingSlots = kLanes * kLaneSlots;
constexpr uint32_t kBurstStop = 0x80000000u;        /* Header::burst: a helper stopped */
constexpr uint32_t kParkDraining = 1u;               /* LaneSignal::park: the helper is releasing this lane's waits */
constexpr uint32_t lane_of(uint32_t slot, uint32_t cls) { return cls * kMaxSlots + slot; }
constexpr uint32_t lane_class(uint32_t lane) { return lane / kMaxSlots; }
constexpr bool lane_is_small(uint32_t lane) { return lane_class(lane) < kSmallClasses; }
constexpr uint32_t kHidden = 2560;
constexpr uint32_t kTopK = 10;
constexpr uint32_t kMaxTopK = 16;     /* capacity bound for per-token scratch (the model's own top-k is checked against it) */
constexpr uint32_t kExperts = 512;
constexpr uint32_t kMaxHeatLdsBytes = 64u * 1024u;   /* the LDS one block has: the per-step route counts (4 bytes per expert) must fit it */
constexpr uint32_t kMaxRows = 4096;
constexpr uint32_t kMaxRoutes = kMaxRows * kTopK;
constexpr uint32_t kActGroup = 128;
constexpr uint32_t kActivationBf16Bytes = kMaxRows * kHidden * sizeof(uint16_t);
constexpr uint32_t kActivationQBytes = kMaxRows * kHidden;
constexpr uint32_t kActivationScaleBytes =
    kMaxRows * (kHidden / kActGroup) * sizeof(float);

struct Route {
    int32_t token;       /* activation/result row to which this expert contribution belongs. */
    int32_t expert;      /* global expert id; addresses the .rad expert group directly. */
    uint16_t weight_bf16;/* router weight exactly as supplied by the patch ABI. */
    uint16_t slot;        /* original top-k slot; gather arithmetic is defined in slot order. */
};
static_assert(sizeof(Route) == 12);

constexpr uint8_t kOwnerLocal = 0xff;

enum class PlacementState : uint32_t {
    ACTIVE=0, PROPOSED=1, RESERVING=2, COPYING=3,
    PREPARED=4, QUIESCING=5, COMMIT=6, RETIRING=7
};
enum class AckPhase : uint32_t { NONE=0, RESERVED=1, READY=2, QUIESCED=3, COMMITTED=4, RETIRED=5, REOPENED=6, CLEANED=7 };
enum class GenerationState : uint32_t { EMPTY=0, STAGING=1, READY=2, ACTIVE=3, RETIRED=4 };

struct alignas(64) PlacementControl {
    uint32_t seq;
    uint32_t state;
    uint32_t active_slot;
    uint32_t pending_slot;
    uint64_t active_epoch;
    uint64_t pending_epoch;
    uint64_t active_digest;
    uint64_t pending_digest;
    uint64_t transaction_id;
    uint64_t deadline_ns;
    uint32_t ready_helpers;
    uint32_t ready_ranks;
    uint32_t quiesced_helpers;
    uint32_t quiesced_ranks;
    uint32_t commit_helpers;
    uint32_t commit_ranks;
    uint32_t retired_helpers;
    uint32_t retired_ranks;
    uint32_t fault_code;
    uint32_t fault_actor;
    uint32_t notify;
    uint32_t expected_helpers;
    uint32_t expected_ranks;
    uint32_t journal_index;
    uint32_t reserved[34];
};
struct alignas(64) GenerationHeader {
    uint64_t epoch;
    uint64_t digest;
    uint32_t map_bytes;
    uint32_t layer_count;
    uint32_t expert_count;
    uint32_t slot;
    uint32_t ready_helpers;
    uint32_t ready_ranks;
    uint32_t state;
    uint32_t reserved[5];
};
struct alignas(64) PlacementAck {
    uint64_t transaction_id;
    uint64_t epoch;
    uint64_t digest;
    uint64_t drained_step;
    uint32_t phase;
    uint32_t generation_slot;
    uint32_t fault_code;
    uint32_t reserved32;
    uint64_t reserved[2];
};
/* Each helper's CPU epoch thread is the sole writer of one coherent two-lane
 * service snapshot. These slots occupy the previously unused ACK page tail.
 * The engine/other helpers never write a foreign helper's slot. */
struct alignas(64) PublishedServiceSlot {
    uint64_t sequence;       /* odd while the helper publishes */
    uint64_t published_ns;   /* CLOCK_MONOTONIC, for rejecting stalled helpers */
    uint64_t samples[2];     /* small and large lanes, never mixed */
    uint64_t rows[2];
    uint64_t time_ns[2];
};
static_assert(sizeof(PublishedServiceSlot)==64);

struct alignas(64) JournalRecord {
    uint64_t transaction_id;
    uint64_t before_epoch;
    uint64_t after_epoch;
    uint64_t before_digest;
    uint64_t candidate_digest;
    uint64_t start_ns;
    uint64_t commit_ns;
    uint64_t finish_ns;
    uint32_t state;
    uint32_t fault;
    uint32_t old_slot;
    uint32_t new_slot;
    uint32_t ready_helpers;
    uint32_t ready_ranks;
    uint32_t commit_helpers;
    uint32_t commit_ranks;
    uint32_t retire_helpers;
    uint32_t retire_ranks;
    // Terminal placement-only cleanup diagnostic, distinct from a precommit
    // abort's original fault; these occupy previously reserved journal words.
    uint32_t cleanup_fault;
    uint32_t cleanup_actor;
    uint32_t reserved[4];
};
struct alignas(64) PlacementJournal {
    JournalRecord entry[2];
};

/* One fixed request slot. Offsets are from the shm base so neither process exchanges pointers. */
struct alignas(64) Request {
    uint32_t sequence;             /* producer publication counter; 0 means never published. */
    uint32_t completion_sequence;  /* helper GPU completion stamp; engine waits for exact sequence. */
    int32_t status;                /* 0 on success, negative errno-style protocol/helper error. */
    uint32_t layer;                /* chooses the expert group in the .rad container. */
    uint32_t slot;                 /* request slot, slot_of(helper, consumer, world): the lane's slot (lane % kMaxSlots). */
    uint32_t world;                /* engine TP ranks, carried per request so a mismatched engine is rejected. */
    uint32_t rows;                 /* number of activation/result rows in this pass. */
    uint32_t hidden;               /* must be 2560; explicit so ABI/model mismatch fails closed. */
    uint32_t route_count;          /* number of Route records valid at route_offset. */
    uint32_t generation_slot;     /* immutable ownership generation, +36: same cache line as selector validation. */
    uint64_t route_offset;         /* routed (token,expert,weight) list for this rank. */
    uint64_t activation_offset;    /* staged BF16 input rows; compare/CPU-reference only. */
    uint64_t activation_q_offset;  /* exact E4M3 operand consumed by the local gate/up GEMM. */
    uint64_t activation_s_offset;  /* exact F32 group scales consumed with activation_q_offset. */
    uint32_t activation_stride;    /* bytes between BF16 reference rows. */
    uint32_t activation_q_stride;  /* bytes between E4M3 rows. */
    uint32_t activation_s_stride;  /* bytes between F32 scale rows. */
    uint32_t result_stride[kMaxEngineRanks]; /* BF16 row stride for each TP rank's result slot. */
    uint64_t result_offset[kResultPlanes]; /* [output rank] planes of this slot; the helper writes only [consumer], full width. */
    /* R3_MOE_WORKER_TRACE: wall_clock64() stamps, each from the clock of the device that wrote it (engine GPU for
     * 0-2, the helper GPU for 3-4; never subtract across devices): 0 publish kernel, 1 wait kernel entered,
     * 2 this request's completion seen by the wait kernel, 3 helper lane began, 4 helper lane published done. On a
     * combined step the engine's full wait runs after the chunked add (RAD42), so 1-2 then measure only the tail. */
    uint64_t trace[5];
    /* R3_MOE_WORKER_COMBINE: nonzero on helper 0's request of a step whose rows helper 0 combines (the generation it then
     * stores in Header::combine.done); written before the request is released, so helper 0 never polls for it. */
    uint32_t combine_generation;
    uint32_t row_base;             /* diagnostic-only pass r0, +204; not a helper selector input. */
    /* R3_MOE_WORKER_TRACE, per request (RAD40): 0 helper lane's activations and routes ready, 1 its gate/up, quant and
     * down done (both wall_clock64 on the helper's clock, with trace[3]/[4] around them); 2 helper 0 only, on the step it
     * combines: microseconds waiting for the other helpers' done words (low 32 bits) and combining (high 32 bits). */
    uint64_t stage_trace[3];
    uint64_t placement_epoch;      /* request epoch, +232. */
    uint64_t completion_epoch;     /* helper completion epoch, +240. */
    uint32_t completion_slot;      /* helper completion generation slot, +248. */
    uint32_t consumer;             /* +252: the engine rank that consumes this request, consumer_of(layer, world) == slot_consumer(slot). */
};
static_assert(sizeof(Request) % 64 == 0);
/* Both engine publisher and helper include this one definition; no older-version compatibility path. */
static_assert(sizeof(Request) == 256, "Request layout changed: bump kVersion");
/* The slot identity a request must carry (RAD49): its lane's slot, the engine width the segment was made for
 * (Header::rank_count, not merely a width the request agrees with itself on), and the consumer the slot and the layer both
 * name. Every reader (helper lanes, CPU reference, combine, replay) checks it the same way. */
/* The same identity without the consumer word, for the helper lane kernels: every thread of a lane kernel validates the request
 * it reads from host-mapped memory, so they read only the Request's first two cache lines (layer, slot and world are +12..+23,
 * as rank/rank_count were before v25); Request::consumer (+252, a fourth line) would add a host-memory read to every thread of
 * every large-lane launch. What the GPU lane check establishes is the EFFECTIVE consumer, slot_consumer(slot, world), and that
 * it is the layer's (consumer_of): the lane writes its result to that rank's plane, chosen from the slot and the segment's
 * width, never from the stored word. The stored Request::consumer word itself is checked only by the CPU readers (combine,
 * host references) through request_slot_valid; a corrupt consumer word alone is therefore not refused by a non-combined GPU
 * lane, but it cannot redirect any result. */
constexpr bool request_lane_identity_valid(uint32_t slot, uint32_t world, uint32_t layer, uint32_t lane_slot,
                                           uint32_t segment_world) {
    return slot == lane_slot && slot < kMaxSlots && world >= 1 && world <= kMaxEngineRanks && world == segment_world &&
           slot_consumer(slot, world) == consumer_of(layer, world);
}
constexpr bool request_slot_valid(uint32_t slot, uint32_t world, uint32_t consumer, uint32_t layer, uint32_t lane_slot,
                                  uint32_t segment_world) {
    return request_lane_identity_valid(slot, world, layer, lane_slot, segment_world) && consumer == slot_consumer(slot, world);
}

struct alignas(64) LaneSignal {
    uint32_t request_sequence; /* engine GPU writes after route/activation data is visible. */
    uint32_t done_sequence;    /* helper GPU writes after the result is visible. */
    uint32_t park;             /* helper CPU: kParkDraining while it releases this lane's waits (RAD20, Header::burst). */
    uint32_t consumed_sequence;   /* engine GPU completion of last result-plane read */
    uint32_t done_slot;
    uint32_t consumed_slot;
    uint64_t done_epoch;
    uint64_t consumed_epoch;
    /* Engine closes admission and retires parents before requesting a drain.
     * Helpers ACK only after their queued waits/compute/combine work drained.
     * Closure IDs prevent an old park from certifying a newer quiescence. */
    uint64_t drain_request;
    uint64_t drain_ack;
    uint32_t reserved[2];
};
static_assert(sizeof(LaneSignal) == 64);
static_assert(offsetof(LaneSignal, drain_request) == 40 && offsetof(LaneSignal, drain_ack) == 48);

/* A helper ACK is a post-drain proof for exactly one closure. The callback
 * owns its stream/CPU-combine drain; failures and stale requests never ACK.
 * Kept card-free so cancellation and stale-proof behavior are unit tested. */
template<class Drain>
inline bool acknowledge_lane_drain(LaneSignal& lane, uint64_t closure, Drain&& drain) {
    if (!closure || __atomic_load_n(&lane.drain_request, __ATOMIC_ACQUIRE) != closure)
        return false;
    if (!drain() || __atomic_load_n(&lane.drain_request, __ATOMIC_ACQUIRE) != closure)
        return false;
    __atomic_store_n(&lane.drain_ack, closure, __ATOMIC_RELEASE);
    return true;
}

/* R3_MOE_WORKER_COMBINE (two to kMaxHelpers helpers, large lane only): helper 0 adds every helper's bf16 rows on the CPU into
 * the consumer rank's combine plane (combine_index) as fp16, so the consumer reads one row a token over its link instead of
 * one a helper. One CombineSignal and one CombineChunks a consumer rank (RAD49): each has one engine writer, and helper 0
 * reads the consumer from its request. The consumer's GPU allocates the helpers' sequences, writes lane/seq/rows here
 * (seq 0: that helper has no request this step) and the generation into helper 0's Request::combine_generation, then
 * releases the requests. The rows go chunk by
 * chunk (CombineChunks, RAD42): helper 0 combines each chunk once every helper has published it and stores chunks.done[c];
 * the engine adds each chunk as it lands and finally waits for `done` = generation and every helper's done word. Not
 * bit-identical to adding the rows on the engine (the sum is rounded to fp16, not bf16). `enabled` is set by helper 0 at
 * start; the engine asks only then. */
constexpr uint32_t kCombineHelpers = kMaxHelpers;   /* the combine adds the rows of up to this many helpers */
/* R3_MOE_WORKER_COMBINE's step shape (RAD49): the consumer asks helper 0 to combine a large-lane step only when helper 0 and at
 * least one other helper have routes in it (a step owned by one helper alone, or by none, keeps the per-helper planes). The
 * publish kernel then also needs CombineSignal::enabled and rows >= min_rows, read from host memory only after this. */
constexpr bool combine_shape_wanted(bool asked, uint32_t helpers, uint32_t cls, bool helper0_routes, bool other_routes) {
    return asked && helpers >= 2 && helpers <= kMaxHelpers && cls == kSmallClasses && helper0_routes && other_routes;
}
struct alignas(64) CombineSignal {
    uint32_t enabled;
    uint32_t min_rows;      /* steps with fewer rows keep the per-helper merge */
    uint32_t request;       /* last generation the engine issued (engine-only counter) */
    uint32_t done;
    uint32_t lane[kCombineHelpers];
    uint32_t seq[kCombineHelpers];   /* 0: no request from that helper this step */
    uint32_t rows;
    uint32_t result_consumed_generation;
    uint32_t result_consumed_slot;
    uint32_t reserved_align;
    uint64_t result_consumed_epoch;
    uint32_t reserved[14];
};
static_assert(sizeof(CombineSignal) == 128);
/* A consumer rank's combine is quiescent when every generation it issued was combined and its result consumed (the engine's
 * helper-drain check, one consumer at a time). */
inline bool combine_quiesced(const CombineSignal& c) {
    const uint32_t request = __atomic_load_n(&c.request, __ATOMIC_ACQUIRE);
    return !request || (__atomic_load_n(&c.done, __ATOMIC_ACQUIRE) == request &&
                        __atomic_load_n(&c.result_consumed_generation, __ATOMIC_ACQUIRE) == request);
}

/* Engine-GPU consumption records, rank-local device memory (never mapped).
 * mailbox_add_finish_kernel updates them every layer without host traffic;
 * the per-parent seal kernel copies changed records into LaneSignal::consumed_*
 * and CombineSignal::result_consumed_* (what the quiescence check reads).
 * `flushed` is the sequence/generation the seal last copied. */
struct ConsumedRecord {
    uint32_t sequence;
    uint32_t slot;
    uint64_t epoch;
    uint32_t flushed;
    uint32_t reserved;
};
static_assert(sizeof(ConsumedRecord) == 24);
struct ConsumedRecords {
    ConsumedRecord lane[kLanes];
    ConsumedRecord combine;
};

/* R3_MOE_WORKER_COMBINE, chunked (RAD42): a large-lane step's rows in chunks of kCombineChunkRows. Helper k's reduce kernel
 * stores its request sequence in seq[k][c] once every row of chunk c is written (system-scope release from its GPU);
 * helper 0's combine waits for seq[k][c] of every helper with a request, sums the chunk and stores the step's
 * generation in done[c]; the engine's add kernel waits for done[c] of a row's chunk before reading it. So the rows
 * written, combined and read overlap chunk by chunk instead of following each other for the whole step (the
 * hand-off measured 1.27x at 256-row chunks, RAD40). */
constexpr uint32_t kCombineChunkRows = 256;
constexpr uint32_t kCombineChunks = kMaxRows / kCombineChunkRows;
struct alignas(64) CombineChunks {
    uint32_t seq[kCombineHelpers][kCombineChunks];   /* helper k's large-lane sequence whose chunk c is written */
    uint32_t done[kCombineChunks];                    /* the combine generation whose chunk c is combined */
};
static_assert(sizeof(CombineChunks) == 320);

/* The engine add kernel's role in a combined step's chunk wait (RAD42). A chunk's first block is the ONLY block that
 * polls done[c] and mirrors it into device memory for the chunk's other blocks, so it must poll whenever the step is
 * combined, whether or not its own token routes to a helper: helper 0 combines every chunk of a combined step, so
 * done[c] always arrives. Gating the poller on its own token's routing deadlocked the engine whenever a chunk's first
 * token was entirely engine-local while a later token of the chunk had a remote expert (the no-tail 8-session hang).
 * constexpr: called from the HIP kernel as well as the host test. */
enum class CombineChunkWait : uint8_t { None, Poll, Mirror };
constexpr CombineChunkWait combine_chunk_wait(bool combined, uint32_t token, bool routed) {
    if (!combined) return CombineChunkWait::None;
    if (token % kCombineChunkRows == 0u) return CombineChunkWait::Poll;
    return routed ? CombineChunkWait::Mirror : CombineChunkWait::None;
}

struct alignas(kPage) Header {
    uint64_t magic;                 /* segment identity. */
    uint32_t version;               /* protocol compatibility gate. */
    uint32_t header_bytes;          /* lets validation reject layout drift explicitly. */
    uint32_t layer_count;           /* model bound; validates request.layer before file access. */
    uint32_t rank_count;            /* the engine's TP ranks (1..kMaxEngineRanks), set by helper 0 from its manifest. */
    uint32_t ring_slots;            /* Request entries following ring_offset. */
    uint32_t max_rows;              /* capacity contract: 4096 activation rows. */
    uint32_t hidden;                /* capacity/model contract: 2560. */
    uint32_t top_k;                 /* capacity contract: 10 routes per token. */
    uint32_t expert_count;          /* model contract: 512 global routed experts. */
    uint32_t route_slot_bytes[kMaxSlots];      /* per-slot routing-list capacity. */
    uint32_t activation_slot_bytes[kMaxSlots]; /* per-slot BF16 reference + exact E4M3 q/F32 scale staging. */
    uint32_t result_slot_bytes[kResultPlanes]; /* [slot][output rank]. */
    uint32_t owned_count;            /* v23 tombstone, must remain zero. */
    uint32_t owner_slot_bytes;       /* v23 tombstone, must remain zero. */
    uint64_t owned_bytes;            /* v23 tombstone, must remain zero. */
    uint64_t owner_offset;           /* v23 tombstone, must remain zero. */
    uint32_t layer_map_slot_bytes;   /* int32_t[kMaxLayers][kExperts], -1 means protected/not regular. */
    uint32_t layer_map_ready[kMaxLayers];    /* engine release-stores after publishing one layer map. */
    uint32_t layer_map_consumed[kMaxLayers]; /* helper release-stores after rebuilding that layer. */
    uint64_t layer_map_offset;
    uint32_t next_sequence;         /* atomic ticket allocator; sequence 0 is reserved. */
    uint32_t publish_sequence;      /* newest published ticket; helper FUTEX_WAITs here. */
    LaneSignal lane[kLanes];          /* GPU-only request/done words, one cache line per lane (lane_of). */
    uint32_t small_class_count;     /* helper-set: enabled small-lane sizes, 1..kSmallClasses. */
    uint32_t small_class_rows[kSmallClasses]; /* helper-set: strictly ascending row limits; a step above the last goes to the large lane. */
    /* Helpers serving this segment (helper 0 sets it). Helper h owns bitmap h and serves the slots slot_of(h, r, world)
     * of every engine rank r; a layer's consumer sends each route to the slot of the helper that owns its expert. */
    uint32_t helper_count;
    uint32_t slot_count;            /* request slots this segment was laid out for (slots_for(helpers, world)); fixed at init, 1..kMaxSlots. */
    /* Set by a helper whose lanes stopped (error or shutdown): an engine still waiting traps at once
     * instead of spinning on a done word that will never move. */
    uint32_t helper_fault;
    uint64_t ring_offset;           /* Request[kRingSlots], page aligned. */
    uint64_t route_offset[kMaxSlots];
    uint64_t activation_offset[kMaxSlots];
    uint64_t result_offset[kResultPlanes]; /* [slot][output rank], like result_slot_bytes. */
    uint64_t segment_bytes;         /* exact mmap/ftruncate/hipHostRegister-able length. */
    uint32_t intermediate;          /* expert intermediate width (RCS09-03): part of the shape both sides must agree on. */
    uint32_t activation_group;      /* activation quantisation group width. */
    uint64_t shape_hash;            /* shape_hash() of the shape fields: a fast agreement check, never the only one. */
    /* RAD20 inference bursts. The engine adds 1 at each burst begin and end (odd = in a burst) and FUTEX_WAKEs; a helper
     * arms its small lanes' pre-queued GPU waits when it turns odd. With power save (R3_MOE_WORKER_POWER_SAVE, default
     * on) no wait is pending between bursts, so the helper GPUs idle (a pending AQL wait keeps a gfx1100 MEC busy at ~95 W, FINDINGS):
     * after a burst ends, a lane sets LaneSignal::park to kParkDraining, re-checks the
     * burst (both sequentially consistent, so the engine's opening add and this flag cannot miss each other), releases
     * and drains its waits, and clears the flag. The engine opens a burst at once and then waits only for lanes still
     * draining, so it never publishes into released waits and never waits for a re-arm. A stopping helper sets
     * kBurstStop and clears its lanes' flags, so no waiter sleeps past the fault. */
    uint32_t burst;
    /* Each helper's process id, written once it has attached (0 before). The engine watches these processes (pidfd)
     * and sets helper_fault when one exits, so a helper killed before it could set helper_fault itself cannot leave
     * the engine's GPU wait spinning (RAD20: an 8.7 h hang after both helper GPUs reset). */
    int32_t helper_pid[kMaxHelpers];
    /* Each helper's process start time (/proc/<pid>/stat field 22), written with helper_pid: the engine checks it after
     * pidfd_open, so a pid reused by another process after the helper died is never watched in its place. */
    uint64_t helper_start[kMaxHelpers];
    /* CPU notification generation: incremented and FUTEX_WAKEd after every change a CPU waiter on this segment waits
     * for (a layer map consumed, a lane park cleared, helper_fault raised). Waiters snapshot it before checking their
     * condition and sleep on it (wait_for), so a change between the check and the sleep is never missed. */
    uint32_t notify;
    /* Bootstrap-only: each helper publishes its final filtered map before helper 0 seals slot 0. */
    uint32_t bootstrap_filtered_helpers;
    uint32_t reserved32[24 - 3 * kMaxHelpers - 2];
    CombineSignal combine[kMaxEngineRanks];   /* [consumer rank] */
    CombineChunks chunks[kMaxEngineRanks];    /* [consumer rank] */
    uint64_t placement_offset;         /* +3712 */
    uint64_t journal_offset;           /* +3720 */
    uint64_t ack_offset;               /* +3728 */
    uint64_t generation_offset[2];     /* +3736, +3744 */
    uint32_t owner_map_bytes;          /* +3752, layer_count * expert_count */
    uint32_t generation_slot_bytes;    /* +3756, page aligned */
    uint32_t placement_control_bytes;  /* +3760 */
    uint32_t placement_ack_count;      /* +3764 */
};
static_assert(kVersion == 25 && kRingSlots == kLanes * kLaneSlots);
static_assert(sizeof(Header) == 4096 && alignof(Header) == 4096);
static_assert(sizeof(Request) == 256 && alignof(Request) == 64);
static_assert(offsetof(Request, generation_slot) == 36);
static_assert(offsetof(Request, row_base) == 204);
static_assert(offsetof(Request, placement_epoch) == 232);
static_assert(offsetof(Request, completion_epoch) == 240);
static_assert(offsetof(Request, completion_slot) == 248);
static_assert(offsetof(Request, consumer) == 252);
/* what request_lane_identity_valid reads stays on the first cache line, with the rest of the lane validation */
static_assert(offsetof(Request, layer) / 64 == 0 && offsetof(Request, slot) / 64 == 0 && offsetof(Request, world) / 64 == 0);
static_assert(sizeof(LaneSignal) == 64 && offsetof(LaneSignal, done_epoch) == 24);
static_assert(offsetof(LaneSignal, consumed_epoch) == 32);
static_assert(sizeof(CombineSignal) == 128);
static_assert(offsetof(CombineSignal, result_consumed_generation) == 52);
static_assert(offsetof(CombineSignal, result_consumed_epoch) == 64);
static_assert(sizeof(PlacementControl) == 256 && alignof(PlacementControl) == 64);
static_assert(offsetof(PlacementControl, active_epoch) == 16);
static_assert(offsetof(PlacementControl, ready_helpers) == 64);
static_assert(offsetof(PlacementControl, notify) == 104);
static_assert(sizeof(GenerationHeader) == 64 && alignof(GenerationHeader) == 64);
static_assert(sizeof(PlacementAck) == 64 && alignof(PlacementAck) == 64);
static_assert(sizeof(JournalRecord) == 128 && sizeof(PlacementJournal) == 256);
static_assert(offsetof(Header, placement_offset) == 3712);
static_assert(offsetof(Header, owner_map_bytes) == 3752);
static_assert(sizeof(PlacementControl) + sizeof(PlacementJournal) +
              (kMaxHelpers + kMaxEngineRanks) * sizeof(PlacementAck) <= kPage);
static_assert(std::is_standard_layout_v<PlacementControl> && std::is_trivially_copyable_v<PlacementControl>);
static_assert(std::is_standard_layout_v<GenerationHeader> && std::is_trivially_copyable_v<GenerationHeader>);
static_assert(std::is_standard_layout_v<PlacementAck> && std::is_trivially_copyable_v<PlacementAck>);
static_assert(std::is_standard_layout_v<PlacementJournal> && std::is_trivially_copyable_v<PlacementJournal>);

constexpr uint64_t align_page(uint64_t n) { return (n + kPage - 1) & ~(uint64_t(kPage) - 1); }
constexpr uint32_t route_slot_bytes() { return kMaxRoutes * sizeof(Route); }
constexpr uint32_t activation_slot_bytes() { return kActivationBf16Bytes + kActivationQBytes + kActivationScaleBytes; }
constexpr uint32_t result_slot_bytes() { return kMaxRows * kHidden * sizeof(uint16_t); }
constexpr uint32_t layer_map_slot_bytes() { return kMaxLayers * kExperts * sizeof(int32_t); }

struct Layout {
    uint64_t ring;
    uint64_t placement, journal, ack, service, generation[2];
    uint64_t layer_map;
    uint64_t route[kMaxSlots];
    uint64_t activation[kMaxSlots];
    uint64_t result[kResultPlanes];
    uint64_t bytes;
    uint32_t route_bytes, activation_bytes, result_bytes, layer_map_bytes;
    uint32_t owner_map_bytes, generation_slot_bytes;
};

/* The same layout from a runtime shape (RCS09-02): every region is sized from the model's hidden width, top-k, expert count, row
 * capacity and activation quantisation group, with checked 64-bit arithmetic, instead of the Flash-Next constants above. The
 * protocol bounds (kMaxSlots, kMaxLayers, the lane ring) stay compile-time. Order and page alignment are exactly layout()'s, so
 * for legacy_shape() the two agree byte for byte (tested). A zero dimension, a zero group, a result that does not fit a 32-bit slot
 * size, or any overflow returns false and leaves *out untouched. */
struct LayoutShape {
    uint32_t hidden, top_k, experts, max_rows, act_group, intermediate;
};
constexpr uint32_t kLegacyIntermediate = 640;   /* Flash-Next's expert width, the only one the execution paths take until RCS09-07 */
constexpr LayoutShape legacy_shape() { return LayoutShape{kHidden, kTopK, kExperts, kMaxRows, kActGroup, kLegacyIntermediate}; }
constexpr bool operator==(const LayoutShape& a, const LayoutShape& b) {
    return a.hidden == b.hidden && a.top_k == b.top_k && a.experts == b.experts && a.max_rows == b.max_rows &&
           a.act_group == b.act_group && a.intermediate == b.intermediate;
}
constexpr bool operator!=(const LayoutShape& a, const LayoutShape& b) { return !(a == b); }
/* FNV-1a over the shape fields: both sides compute it from their own model and compare it first; the fields are compared too. */
constexpr uint64_t shape_hash(const LayoutShape& s) {
    uint64_t h = 1469598103934665603ull;
    const uint32_t v[] = {s.hidden, s.top_k, s.experts, s.max_rows, s.act_group, s.intermediate};
    for (uint32_t x : v)
        for (int i = 0; i < 4; ++i) { h ^= (x >> (8 * i)) & 0xffu; h *= 1099511628211ull; }
    return h;
}

/* Checked runtime layout; all generation map storage uses actual layer and expert counts. */
inline bool layout_for(const LayoutShape& s, uint32_t layer_count, uint32_t slots, Layout* out) {
    auto mul = [](uint64_t a, uint64_t b, uint64_t* r) {
        if (a && b > UINT64_MAX / a) return false;
        *r = a * b;
        return true;
    };
    auto add = [](uint64_t a, uint64_t b, uint64_t* r) {
        if (b > UINT64_MAX - a) return false;
        *r = a + b;
        return true;
    };
    auto page = [&](uint64_t n, uint64_t* r) {
        uint64_t t = 0;
        if (!add(n, kPage - 1, &t)) return false;
        *r = t & ~(uint64_t(kPage) - 1);
        return true;
    };
    if (!out || layer_count == 0 || layer_count > kMaxLayers ||
        slots < 1 || slots > kMaxSlots ||
        !s.hidden || !s.top_k || !s.experts || !s.max_rows || !s.act_group ||
        !s.intermediate || s.top_k > s.experts) return false;
    uint64_t routes = 0, rows_hidden = 0, groups = 0;
    uint64_t route_b = 0, bf16_b = 0, q_b = 0, scale_b = 0, act_b = 0;
    uint64_t result_b = 0, map_b = 0, owner_map_b = 0, generation_b = 0, tmp = 0;
    if (!mul(s.max_rows, s.top_k, &routes) || !mul(routes, sizeof(Route), &route_b)) return false;
    if (!mul(s.max_rows, s.hidden, &rows_hidden) || !mul(rows_hidden, sizeof(uint16_t), &bf16_b)) return false;
    q_b = rows_hidden;
    groups = (uint64_t)s.hidden / s.act_group;
    if (!groups || groups * s.act_group != s.hidden) return false;
    if (!mul(s.max_rows, groups, &tmp) || !mul(tmp, sizeof(float), &scale_b)) return false;
    if (!add(bf16_b, q_b, &tmp) || !add(tmp, scale_b, &act_b)) return false;
    result_b = bf16_b;
    if (!mul(layer_count, s.experts, &owner_map_b) ||
        !mul((uint64_t)kMaxLayers * s.experts, sizeof(int32_t), &map_b) ||
        !add(sizeof(GenerationHeader), owner_map_b, &tmp) || !page(tmp, &generation_b)) return false;
    for (uint64_t v : {route_b, act_b, result_b, map_b, owner_map_b, generation_b})
        if (v > UINT32_MAX) return false;
    Layout x{};
    uint64_t at = kPage;
    x.ring = at;
    if (!add(at, (uint64_t)sizeof(Request) * kRingSlots, &tmp) || !page(tmp, &at)) return false;
    x.placement = at;
    if (!add(at, sizeof(PlacementControl), &at)) return false;
    x.journal = at;
    if (!add(at, sizeof(PlacementJournal), &at)) return false;
    x.ack = at;
    if (!add(at, (kMaxHelpers + kMaxEngineRanks) * sizeof(PlacementAck), &x.service) ||
        !add(x.service, kMaxHelpers * sizeof(PublishedServiceSlot), &tmp) ||
        !page(tmp, &at)) return false;
    x.generation[0] = at;
    if (!add(at, generation_b, &at)) return false;
    x.generation[1] = at;
    if (!add(at, generation_b, &at)) return false;
    x.layer_map = at;
    if (!add(at, map_b, &tmp) || !page(tmp, &at)) return false;
    for (uint32_t r = 0; r < slots; ++r) {
        x.route[r] = at;
        if (!add(at, route_b, &tmp) || !page(tmp, &at)) return false;
        x.activation[r] = at;
        if (!add(at, act_b, &tmp) || !page(tmp, &at)) return false;
        for (uint32_t o = 0; o < kMaxEngineRanks; ++o) {
            x.result[result_index(r, o)] = at;
            if (!add(at, result_b, &tmp) || !page(tmp, &at)) return false;
        }
    }
    if (!page(at, &x.bytes) || x.bytes > SIZE_MAX) return false;
    x.route_bytes = (uint32_t)route_b; x.activation_bytes = (uint32_t)act_b;
    x.result_bytes = (uint32_t)result_b; x.layer_map_bytes = (uint32_t)map_b;
    x.owner_map_bytes = (uint32_t)owner_map_b;
    x.generation_slot_bytes = (uint32_t)generation_b;
    *out = x;
    return true;
}

/* FNV-1a 64 over explicitly little-endian fields followed by exactly the canonical map. */
inline uint64_t placement_digest(uint64_t model_shape_hash, uint32_t layers,
                                 uint32_t experts, uint32_t helpers, const uint8_t* owners) {
    uint64_t hash = 14695981039346656037ull;
    auto u8 = [&](uint8_t byte) { hash = (hash ^ byte) * 1099511628211ull; };
    auto u32 = [&](uint32_t value) {
        for (uint32_t i = 0; i != 4; ++i) u8((uint8_t)(value >> (i * 8)));
    };
    for (uint32_t i = 0; i != 8; ++i) u8((uint8_t)(model_shape_hash >> (i * 8)));
    u32(layers); u32(experts); u32(helpers);
    if (owners)
        for (uint64_t i = 0, n = (uint64_t)layers * experts; i < n; ++i) u8(owners[i]);
    return hash;
}

/* Lay out and stamp a segment for this shape. The caller has mapped at least layout_for(shape).bytes. False (and nothing written)
 * when the shape does not lay out. */
inline bool init(void* base, uint32_t layers, const LayoutShape& shape, uint32_t helper_count, uint32_t engine_ranks) {
    Layout l{};
    if (!base || !slots_fit(helper_count, engine_ranks) ||
        !layout_for(shape, layers, slots_for(helper_count, engine_ranks), &l)) return false;
    std::memset(base, 0, static_cast<size_t>(l.bytes));
    auto* h = static_cast<Header*>(base);
    h->magic = kMagic; h->version = kVersion; h->header_bytes = sizeof(Header);
    h->layer_count = layers; h->rank_count = engine_ranks; h->slot_count = slots_for(helper_count, engine_ranks);
    h->ring_slots = kRingSlots;
    h->max_rows = shape.max_rows; h->hidden = shape.hidden; h->top_k = shape.top_k; h->expert_count = shape.experts;
    h->intermediate = shape.intermediate; h->activation_group = shape.act_group; h->shape_hash = shape_hash(shape);
    /* helper_count remains unpublished until helper 0 finishes its bootstrap metadata. */
    h->layer_map_slot_bytes = l.layer_map_bytes; h->layer_map_offset = l.layer_map;
    std::memset(static_cast<uint8_t*>(base) + l.layer_map, 0xff, l.layer_map_bytes);
    h->next_sequence = 1; h->ring_offset = l.ring; h->segment_bytes = l.bytes;
    h->small_class_count = 1; h->small_class_rows[0] = 8;
    h->placement_offset = l.placement; h->journal_offset = l.journal; h->ack_offset = l.ack;
    h->generation_offset[0] = l.generation[0]; h->generation_offset[1] = l.generation[1];
    h->owner_map_bytes = l.owner_map_bytes; h->generation_slot_bytes = l.generation_slot_bytes;
    h->placement_control_bytes = sizeof(PlacementControl);
    h->placement_ack_count = kMaxHelpers + kMaxEngineRanks;
    std::memset(static_cast<uint8_t*>(base) + l.service, 0,
                kMaxHelpers * sizeof(PublishedServiceSlot));
    for (uint32_t slot = 0; slot < 2; ++slot) {
        auto* g = reinterpret_cast<GenerationHeader*>(static_cast<uint8_t*>(base) + l.generation[slot]);
        g->map_bytes = l.owner_map_bytes; g->layer_count = layers;
        g->expert_count = shape.experts; g->slot = slot;
        g->state = (uint32_t)(slot == 0 ? GenerationState::ACTIVE : GenerationState::EMPTY);
        std::memset(reinterpret_cast<uint8_t*>(g) + sizeof(GenerationHeader), kOwnerLocal, l.owner_map_bytes);
    }
    auto* g0 = reinterpret_cast<GenerationHeader*>(static_cast<uint8_t*>(base) + l.generation[0]);
    g0->epoch = 1;
    g0->digest = placement_digest(h->shape_hash, layers, shape.experts, helper_count,
                                  reinterpret_cast<uint8_t*>(g0) + sizeof(GenerationHeader));
    auto* control = reinterpret_cast<PlacementControl*>(static_cast<uint8_t*>(base) + l.placement);
    control->state = (uint32_t)PlacementState::ACTIVE;
    control->active_slot = 0; control->pending_slot = UINT32_MAX;
    control->active_epoch = 1; control->active_digest = g0->digest;
    control->expected_helpers = (1u << helper_count) - 1u;
    for (uint32_t r = 0; r < h->slot_count; ++r) {
        h->route_slot_bytes[r] = l.route_bytes;
        h->activation_slot_bytes[r] = l.activation_bytes;
        h->route_offset[r] = l.route[r];
        /* Every helper reads the consumer's step activations: they are staged once, in helper 0's slot for that rank. */
        h->activation_offset[r] = l.activation[slot_of(0, slot_consumer(r, engine_ranks), engine_ranks)];
        for (uint32_t o = 0; o < kMaxEngineRanks; ++o) {
            const uint32_t i = result_index(r, o);
            h->result_slot_bytes[i] = l.result_bytes;
            h->result_offset[i] = l.result[i];
        }
    }
    return true;
}

/* A mapped segment is acceptable when it is a kVersion segment of exactly the shape this side expects, and its layout is the one that
 * shape implies: every offset, slot size and length is reconstructed from the header's own shape fields, so a corrupt or foreign
 * header cannot point a request outside the mapping. A different shape, a stale protocol version or a malformed layout is refused
 * before any request is published or any expert buffer is touched. */
inline bool valid(const Header* h, size_t mapped, const LayoutShape& expect) {
    if (!h || h->magic != kMagic || h->version != kVersion || h->header_bytes != sizeof(Header) ||
        !slots_fit(h->helper_count, h->rank_count) ||
        h->slot_count != slots_for(h->helper_count, h->rank_count) || h->ring_slots != kRingSlots ||
        h->layer_count == 0 || h->layer_count > kMaxLayers ||
        h->small_class_count == 0 || h->small_class_count > kSmallClasses ||
        h->owned_count || h->owner_slot_bytes || h->owned_bytes || h->owner_offset) return false;
    const LayoutShape got{h->hidden, h->top_k, h->expert_count, h->max_rows,
                          h->activation_group, h->intermediate};
    if (got != expect || h->shape_hash != shape_hash(got)) return false;
    Layout l{};
    if (!layout_for(got, h->layer_count, h->slot_count, &l) ||
        h->ring_offset != l.ring || h->layer_map_offset != l.layer_map ||
        h->layer_map_slot_bytes != l.layer_map_bytes ||
        h->placement_offset != l.placement || h->journal_offset != l.journal ||
        h->ack_offset != l.ack || h->generation_offset[0] != l.generation[0] ||
        h->generation_offset[1] != l.generation[1] ||
        h->owner_map_bytes != l.owner_map_bytes ||
        h->generation_slot_bytes != l.generation_slot_bytes ||
        h->placement_control_bytes != sizeof(PlacementControl) ||
        h->placement_ack_count != kMaxHelpers + kMaxEngineRanks ||
        h->segment_bytes != l.bytes || mapped < l.bytes) return false;
    if (h->bootstrap_filtered_helpers & ~((1u << h->helper_count) - 1u))
        return false;
    for (uint32_t v : h->reserved32) if (v) return false;
    for (uint32_t c = 0; c < h->small_class_count; ++c)
        if (h->small_class_rows[c] == 0 || h->small_class_rows[c] > h->max_rows ||
            (c && h->small_class_rows[c] <= h->small_class_rows[c - 1])) return false;
    const auto* base = reinterpret_cast<const uint8_t*>(h);
    const auto* control = reinterpret_cast<const PlacementControl*>(base + l.placement);
    if (control->active_slot > 1 || (control->pending_slot > 1 && control->pending_slot != UINT32_MAX) ||
        !control->active_epoch || control->state > (uint32_t)PlacementState::RETIRING ||
        control->expected_helpers != (1u << h->helper_count) - 1u ||
        (control->state == (uint32_t)PlacementState::ACTIVE &&
         (control->pending_epoch || control->pending_digest || control->pending_slot != UINT32_MAX)))
        return false;
    for (uint32_t v : control->reserved) if (v) return false;
    for (uint32_t slot = 0; slot < 2; ++slot) {
        const auto* g = reinterpret_cast<const GenerationHeader*>(base + l.generation[slot]);
        if (g->map_bytes != l.owner_map_bytes || g->layer_count != h->layer_count ||
            g->expert_count != h->expert_count || g->slot != slot ||
            g->state > (uint32_t)GenerationState::RETIRED) return false;
        for (uint32_t v : g->reserved) if (v) return false;
        if (g->state == (uint32_t)GenerationState::EMPTY && (g->epoch || g->digest))
            return false;
        if (slot == control->active_slot) {
            const uint8_t* map = reinterpret_cast<const uint8_t*>(g) + sizeof(GenerationHeader);
            for (uint32_t i = 0; i < g->map_bytes; ++i) {
                const uint8_t owner = __atomic_load_n(map + i, __ATOMIC_ACQUIRE);
                if (owner != kOwnerLocal && owner >= h->helper_count) return false;
            }
            if (g->epoch != control->active_epoch || g->digest != control->active_digest ||
                g->state != (uint32_t)GenerationState::ACTIVE) return false;
            /* The initial map is mutable during bootstrap; READY helper mask seals it. */
            if (g->ready_helpers == control->expected_helpers &&
                placement_digest(h->shape_hash, h->layer_count, h->expert_count, h->helper_count,
                                 reinterpret_cast<const uint8_t*>(g) + sizeof(GenerationHeader)) != g->digest)
                return false;
        }
    }
    for (uint32_t r = 0; r < kMaxSlots; ++r) {
        const bool used = r < h->slot_count;
        /* Every helper's slot for a rank shares helper 0's activation staging for that rank. */
        const uint32_t home = used ? slot_of(0, slot_consumer(r, h->rank_count), h->rank_count) : 0u;
        if (h->route_offset[r] != (used ? l.route[r] : 0) ||
            h->activation_offset[r] != (used ? l.activation[home] : 0) ||
            h->route_slot_bytes[r] != (used ? l.route_bytes : 0) ||
            h->activation_slot_bytes[r] != (used ? l.activation_bytes : 0)) return false;
        for (uint32_t o = 0; o < kMaxEngineRanks; ++o) {
            const uint32_t i = result_index(r, o);
            if (h->result_offset[i] != (used ? l.result[i] : 0) ||
                h->result_slot_bytes[i] != (used ? l.result_bytes : 0)) return false;
        }
    }
    return true;
}

/* Validate every descriptor and version before any GPU registration. */
inline bool valid_self(const Header* h, size_t mapped) {
    if (!h || h->version != kVersion) return false;
    return valid(h, mapped, LayoutShape{h->hidden, h->top_k, h->expert_count,
                                       h->max_rows, h->activation_group, h->intermediate});
}

template <class T> inline T* ptr(void* base, uint64_t off) {
    return reinterpret_cast<T*>(static_cast<uint8_t*>(base) + off);
}
template <class T> inline const T* ptr(const void* base, uint64_t off) {
    return reinterpret_cast<const T*>(static_cast<const uint8_t*>(base) + off);
}
inline Request* request(void* base, uint32_t seq) {
    auto* h = static_cast<Header*>(base);
    return ptr<Request>(base, h->ring_offset) + ((seq - 1) % kRingSlots);
}
inline const Request* request(const void* base, uint32_t seq) {
    auto* h = static_cast<const Header*>(base);
    return ptr<Request>(base, h->ring_offset) + ((seq - 1) % kRingSlots);
}

/* The lane class a step of `rows` rows takes: the first small class that fits it, else the large lane (kSmallClasses). */
inline uint32_t step_class(const Header* h, uint32_t rows) {
    for (uint32_t c = 0; c < h->small_class_count && c < kSmallClasses; ++c)
        if (rows <= h->small_class_rows[c]) return c;
    return kSmallClasses;
}

inline Request* lane_request(void* base, uint32_t lane, uint32_t seq) {
    auto* h = static_cast<Header*>(base);
    if (lane >= kLanes || seq == 0) return nullptr;
    return ptr<Request>(base, h->ring_offset) + lane * kLaneSlots + ((seq - 1) % kLaneSlots);
}
inline const Request* lane_request(const void* base, uint32_t lane, uint32_t seq) {
    auto* h = static_cast<const Header*>(base);
    if (lane >= kLanes || seq == 0) return nullptr;
    return ptr<Request>(base, h->ring_offset) + lane * kLaneSlots + ((seq - 1) % kLaneSlots);
}

inline PlacementControl* placement_control(void* base) {
    auto* h = static_cast<Header*>(base);
    return ptr<PlacementControl>(base, h->placement_offset);
}
inline const PlacementControl* placement_control(const void* base) {
    auto* h = static_cast<const Header*>(base);
    return ptr<PlacementControl>(base, h->placement_offset);
}
/* Bounded seqlock snapshot: a writer publishes odd seq, data, even seq with release.
 * No coordinator writes these words after initialization in slice A. */
inline bool placement_snapshot(const void* base, PlacementControl* out) {
    if (!base || !out) return false;
    const PlacementControl* c = placement_control(base);
    for (uint32_t tries = 0; tries < 64; ++tries) {
        const uint32_t before = __atomic_load_n(&c->seq, __ATOMIC_ACQUIRE);
        if (before & 1u) continue;
        std::memcpy(out, c, sizeof(*out));
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        if (before == __atomic_load_n(&c->seq, __ATOMIC_ACQUIRE)) return true;
    }
    return false;
}
inline PlacementJournal* placement_journal(void* base) {
    auto* h = static_cast<Header*>(base);
    return ptr<PlacementJournal>(base, h->journal_offset);
}
inline const PlacementJournal* placement_journal(const void* base) {
    auto* h = static_cast<const Header*>(base);
    return ptr<PlacementJournal>(base, h->journal_offset);
}
/* The v24 service slots are part of the control page, directly after ACKs.
 * No extra mapping or dynamic allocator; layout_for validates the page fit. */
inline PublishedServiceSlot* published_service(void* base, uint32_t helper) {
    auto* h = static_cast<Header*>(base);
    return h && helper < h->helper_count && helper < kMaxHelpers
        ? ptr<PublishedServiceSlot>(base, h->ack_offset +
                (kMaxHelpers + kMaxEngineRanks) * sizeof(PlacementAck)) + helper : nullptr;
}
inline const PublishedServiceSlot* published_service(const void* base, uint32_t helper) {
    return published_service(const_cast<void*>(base), helper);
}

inline PlacementAck* placement_ack(void* base, uint32_t actor) {
    auto* h = static_cast<Header*>(base);
    return actor < h->placement_ack_count ? ptr<PlacementAck>(base, h->ack_offset) + actor : nullptr;
}
inline const PlacementAck* placement_ack(const void* base, uint32_t actor) {
    auto* h = static_cast<const Header*>(base);
    return actor < h->placement_ack_count ? ptr<PlacementAck>(base, h->ack_offset) + actor : nullptr;
}
inline GenerationHeader* generation_header(void* base, uint32_t slot) {
    auto* h = static_cast<Header*>(base);
    return slot < 2 ? ptr<GenerationHeader>(base, h->generation_offset[slot]) : nullptr;
}
inline const GenerationHeader* generation_header(const void* base, uint32_t slot) {
    auto* h = static_cast<const Header*>(base);
    return slot < 2 ? ptr<GenerationHeader>(base, h->generation_offset[slot]) : nullptr;
}
inline uint8_t* generation_map(void* base, uint32_t slot) {
    auto* g = generation_header(base, slot);
    return g ? reinterpret_cast<uint8_t*>(g) + sizeof(GenerationHeader) : nullptr;
}
inline const uint8_t* generation_map(const void* base, uint32_t slot) {
    auto* g = generation_header(base, slot);
    return g ? reinterpret_cast<const uint8_t*>(g) + sizeof(GenerationHeader) : nullptr;
}
inline uint8_t owner_at(const void* base, uint32_t slot, uint32_t layer, uint32_t expert) {
    const auto* h = static_cast<const Header*>(base);
    const uint8_t* map = h ? generation_map(base, slot) : nullptr;
    if (!map || layer >= h->layer_count || expert >= h->expert_count) return kOwnerLocal;
    return __atomic_load_n(map + (size_t)layer * h->expert_count + expert, __ATOMIC_ACQUIRE);
}
inline uint64_t generation_digest(const void* base, uint32_t slot) {
    const auto* h = static_cast<const Header*>(base);
    const uint8_t* map = h ? generation_map(base, slot) : nullptr;
    return map ? placement_digest(h->shape_hash, h->layer_count, h->expert_count,
                                  h->helper_count, map) : 0;
}

/* The bootstrap publication is a distinct controller seqlock transition.
 * The initial init() digest alone is not a seal, and must never be uploaded. */
inline bool bootstrap_generation_sealed(const void* base) {
    if (!base) return false;
    const auto* h = static_cast<const Header*>(base);
    PlacementControl c{};
    if (!placement_snapshot(base, &c) || c.seq < 2 || (c.seq & 1u) ||
        !h->helper_count || h->helper_count > kMaxHelpers ||
        c.active_slot != 0 || c.active_epoch != 1 || c.fault_code ||
        c.expected_helpers != ((1u << h->helper_count) - 1u) ||
        __atomic_load_n(&h->bootstrap_filtered_helpers, __ATOMIC_ACQUIRE) != c.expected_helpers)
        return false;
    const auto* g = generation_header(base, 0);
    return g && g->state == uint32_t(GenerationState::ACTIVE) && g->epoch == 1 &&
           __atomic_load_n(&g->ready_helpers, __ATOMIC_ACQUIRE) == c.expected_helpers &&
           g->digest == c.active_digest && c.active_digest &&
           generation_digest(base, 0) == c.active_digest;
}

inline int32_t* layer_slots(void* base, uint32_t layer) {
    auto* h = static_cast<Header*>(base);
    if (!h || layer >= h->layer_count) return nullptr;
    return ptr<int32_t>(base, h->layer_map_offset) + (size_t)layer * h->expert_count;
}
inline const int32_t* layer_slots(const void* base, uint32_t layer) {
    auto* h = static_cast<const Header*>(base);
    if (!h || layer >= h->layer_count) return nullptr;
    return ptr<int32_t>(base, h->layer_map_offset) + (size_t)layer * h->expert_count;
}
inline int32_t layer_slot(const void* base, uint32_t layer, uint32_t expert) {
    const int32_t* p = layer_slots(base, layer);
    return p && expert < static_cast<const Header*>(base)->expert_count ? p[expert] : -1;
}
/* Sequence order modulo 2^32: a is after b when it is less than half the space ahead (defined unsigned arithmetic, no
 * signed cast). Holds while fewer than 2^31 sequences are in flight between the two, which the protocol keeps. */
constexpr bool seq_after(uint32_t a, uint32_t b) { return a != b && (uint32_t)(a - b) < 0x80000000u; }

inline uint32_t load(const uint32_t* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
inline void store(uint32_t* p, uint32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
inline uint32_t fetch_add(uint32_t* p, uint32_t v) { return __atomic_fetch_add(p, v, __ATOMIC_ACQ_REL); }
inline int futex_wait(uint32_t* p, uint32_t expected) {
    return static_cast<int>(::syscall(SYS_futex, p, FUTEX_WAIT, expected, nullptr, nullptr, 0));
}
/* Every waiter: two helpers (and the engine) can wait on the same word of a shared segment. */
inline int futex_wake(uint32_t* p) {
    return static_cast<int>(::syscall(SYS_futex, p, FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0));
}
/* Header::notify: publish a change CPU waiters wait for (after the change itself is stored). */
inline void notify(Header* h) {
    (void)__atomic_fetch_add(&h->notify, 1u, __ATOMIC_SEQ_CST);
    (void)futex_wake(&h->notify);
}

/* Helper drain (after the lane's stream is drained): publish the identity of the lane's last completed request
 * (epoch/slot) for the engine's quiescence check. The ring slot must still hold that request (admission closed and
 * every parent retired, so nothing newer was published) and carry a valid epoch/slot; otherwise refuse the ACK
 * rather than copy another request's identity. acknowledge_lane_drain's release of drain_ack publishes the words. */
inline bool publish_done_identity(void* base, uint32_t lane) {
    LaneSignal& l = static_cast<Header*>(base)->lane[lane];
    const uint32_t seq = load(&l.done_sequence);
    if (!seq) return true;
    const Request* q = lane_request(base, lane, seq);
    if (!q || load(&q->sequence) != seq) return false;
    const uint64_t epoch = __atomic_load_n(&q->placement_epoch, __ATOMIC_ACQUIRE);
    const uint32_t slot = load(&q->generation_slot);
    if (!epoch || slot > 1) return false;
    __atomic_store_n(&l.done_epoch, epoch, __ATOMIC_RELAXED);
    __atomic_store_n(&l.done_slot, slot, __ATOMIC_RELAXED);
    return true;
}

/* Engine quiescence, one lane: a served lane acknowledged exactly this closure, every published request is done and
 * consumed, and the helper's done identity equals the engine's consumed identity for the last one. */
inline bool lane_quiesced(const Header* h, uint32_t lane, uint64_t closure, bool served) {
    const LaneSignal& l = h->lane[lane];
    if (served && __atomic_load_n(&l.drain_ack, __ATOMIC_ACQUIRE) != closure) return false;
    const uint32_t seq = load(&l.request_sequence);
    if (load(&l.done_sequence) != seq || load(&l.consumed_sequence) != seq) return false;
    return !seq || (__atomic_load_n(&l.done_epoch, __ATOMIC_ACQUIRE) == __atomic_load_n(&l.consumed_epoch, __ATOMIC_ACQUIRE) &&
                    load(&l.done_slot) == load(&l.consumed_slot));
}
/* Raise helper_fault (sticky) and wake every CPU waiter. */
inline void raise_fault(Header* h) {
    __atomic_store_n(&h->helper_fault, 1u, __ATOMIC_SEQ_CST);
    notify(h);
}

enum class WaitResult { Ready, Fault, Error, Timeout };
/* Sleep until done() holds, or until `deadline` (CLOCK_MONOTONIC; nullptr: none) passes. helper_fault wins over done():
 * once a helper has failed, nothing it published since counts. The generation is read before both checks, so a notify()
 * between the checks and the sleep makes the futex return EAGAIN instead of sleeping (no lost wake-up). */
template <class Done>
inline WaitResult wait_for(Header* h, const timespec* deadline, Done done) {
    for (;;) {
        const uint32_t gen = __atomic_load_n(&h->notify, __ATOMIC_SEQ_CST);
        if (__atomic_load_n(&h->helper_fault, __ATOMIC_SEQ_CST)) return WaitResult::Fault;
        if (done()) return WaitResult::Ready;
        timespec left{}, *rel = nullptr;
        if (deadline) {
            timespec now{};
            clock_gettime(CLOCK_MONOTONIC, &now);
            int64_t ns = (int64_t)(deadline->tv_sec - now.tv_sec) * 1000000000 + (deadline->tv_nsec - now.tv_nsec);
            if (ns <= 0) return WaitResult::Timeout;
            left = {(time_t)(ns / 1000000000), (long)(ns % 1000000000)};
            rel = &left;
        }
        if (::syscall(SYS_futex, &h->notify, FUTEX_WAIT, gen, rel, nullptr, 0) < 0 && errno != EAGAIN &&
            errno != EINTR && errno != ETIMEDOUT)
            return WaitResult::Error;
    }
}

/* A process's start time in clock ticks (/proc/<pid>/stat field 22), 0 when it cannot be read. With the pid it names
 * one process for the life of the system. */
inline uint64_t process_start_ticks(int pid) {
    char path[64], buf[1024];
    std::snprintf(path, sizeof(path), "/proc/%d/stat", pid);
    FILE* f = std::fopen(path, "r");
    if (!f) return 0;
    const size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
    std::fclose(f);
    buf[n] = '\0';
    const char* p = std::strrchr(buf, ')');   /* comm may contain spaces; the fields after it do not */
    if (!p) return 0;
    for (int field = 2; field < 22 && *p; ++p)
        if (*p == ' ') ++field;
    return std::strtoull(p, nullptr, 10);
}

/* Sleep until *p == value: true then; false when *fault becomes non-zero (whoever sets it wakes p) or the futex
 * fails, so a failed wait is never taken for the value having arrived. */
inline bool wait_until(uint32_t* p, uint32_t value, const uint32_t* fault) {
    for (;;) {
        const uint32_t seen = load(p);
        if (seen == value) return true;
        if (load(fault)) return false;
        if (futex_wait(p, seen) < 0 && errno != EAGAIN && errno != EINTR) return false;
    }
}

} // namespace r3::moe_shm
