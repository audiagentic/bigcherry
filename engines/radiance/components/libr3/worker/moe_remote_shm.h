#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <climits>
#include <cstring>
#include <initializer_list>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace r3::moe_shm {

constexpr uint64_t kMagic = 0x31454f4d33485253ull; /* "SRH3MOE1": reject unrelated/stale shm. */
constexpr uint32_t kVersion = 14;  /* 14: the header records the model shape (intermediate, activation_group, shape_hash) and every region is laid out from it (layout_for); 13: several pre-queued small-lane sizes (Header::small_class_*), ten lanes; 12: Request carries the R3_MOE_WORKER_TRACE stamps (192 -> 256 bytes); 11: helper_fault; completion is an exact generation */
constexpr uint32_t kPage = 4096;
constexpr uint32_t kMaxRanks = 2;       /* Flash-Next configuration of record: TP=2. */
constexpr uint32_t kMaxLayers = 128;     /* ownership bitmap bound; Flash-Next is below this. */
/* Lanes a producer rank (RAD12): up to kSmallClasses small lanes, each pre-queued by the helper at its own fixed row
 * count (Header::small_class_rows, ascending; a step goes to the first class that fits its rows), and one large lane
 * for the larger (prefill) steps, which the helper launches sized to the request when it arrives. A rank publishes one
 * request at a time, so every lane shares that rank's route/activation/result slots. Lane index = class * kMaxRanks + rank. */
constexpr uint32_t kSmallClasses = 4;
constexpr uint32_t kClasses = kSmallClasses + 1;          /* the last class is the large lane */
constexpr uint32_t kLanes = kMaxRanks * kClasses;
constexpr uint32_t kLaneSlots = 4;                        /* lane-private metadata ring; bulk slots stay per-rank. */
constexpr uint32_t kRingSlots = kLanes * kLaneSlots;
constexpr uint32_t lane_of(uint32_t rank, uint32_t cls) { return cls * kMaxRanks + rank; }
constexpr uint32_t lane_class(uint32_t lane) { return lane / kMaxRanks; }
constexpr bool lane_is_small(uint32_t lane) { return lane_class(lane) < kSmallClasses; }
constexpr uint32_t kHidden = 2560;
constexpr uint32_t kTopK = 10;
constexpr uint32_t kExperts = 512;
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

/* One fixed request slot. Offsets are from the shm base so neither process exchanges pointers. */
struct alignas(64) Request {
    uint32_t sequence;             /* producer publication counter; 0 means never published. */
    uint32_t completion_sequence;  /* helper stores sequence here, then FUTEX_WAKEs waiters. */
    int32_t status;                /* 0 on success, negative errno-style protocol/helper error. */
    uint32_t layer;                /* chooses the expert group in the .rad container. */
    uint32_t rank;                 /* rank whose TP slice this helper must compute. */
    uint32_t rank_count;           /* carried per request so a mismatched engine is rejected. */
    uint32_t rows;                 /* number of activation/result rows in this pass. */
    uint32_t hidden;               /* must be 2560; explicit so ABI/model mismatch fails closed. */
    uint32_t route_count;          /* number of Route records valid at route_offset. */
    uint32_t row_base;             /* pass r0 in the step; diagnostic only. */
    uint64_t route_offset;         /* routed (token,expert,weight) list for this rank. */
    uint64_t activation_offset;    /* staged BF16 input rows; compare/CPU-reference only. */
    uint64_t activation_q_offset;  /* exact E4M3 operand consumed by the local gate/up GEMM. */
    uint64_t activation_s_offset;  /* exact F32 group scales consumed with activation_q_offset. */
    uint32_t activation_stride;    /* bytes between BF16 reference rows. */
    uint32_t activation_q_stride;  /* bytes between E4M3 rows. */
    uint32_t activation_s_stride;  /* bytes between F32 scale rows. */
    uint32_t result_stride[kMaxRanks]; /* BF16 row stride for each TP rank's result slot. */
    uint64_t result_offset[kMaxRanks * kMaxRanks]; /* partial-output slots; helper writes only req.rank. */
    /* R3_MOE_WORKER_TRACE: wall_clock64() stamps, each from the clock of the device that wrote it (engine GPU for
     * 0-2, the helper GPU for 3-4; never subtract across devices): 0 publish kernel, 1 wait kernel entered,
     * 2 this request's completion seen by the wait kernel, 3 helper lane began, 4 helper lane published done. */
    uint64_t trace[10];
};
static_assert(sizeof(Request) % 64 == 0);
/* The ring ends on the same page boundary for 192- and 256-byte requests, so the header's size checks cannot tell two
 * layouts apart: a layout change must bump kVersion, and this fires so it cannot be forgotten. */
static_assert(sizeof(Request) == 256, "Request layout changed: bump kVersion");

struct alignas(64) LaneSignal {
    uint32_t request_sequence; /* engine GPU writes after route/activation data is visible. */
    uint32_t done_sequence;    /* helper GPU writes after the result is visible. */
    uint32_t reserved[14];
};
static_assert(sizeof(LaneSignal) == 64);

struct alignas(kPage) Header {
    uint64_t magic;                 /* segment identity. */
    uint32_t version;               /* protocol compatibility gate. */
    uint32_t header_bytes;          /* lets validation reject layout drift explicitly. */
    uint32_t layer_count;           /* model bound; validates request.layer before file access. */
    uint32_t rank_count;            /* configured TP ranks; fixed to 2 for this slice. */
    uint32_t ring_slots;            /* Request entries following ring_offset. */
    uint32_t max_rows;              /* capacity contract: 4096 activation rows. */
    uint32_t hidden;                /* capacity/model contract: 2560. */
    uint32_t top_k;                 /* capacity contract: 10 routes per token. */
    uint32_t expert_count;          /* model contract: 512 global routed experts. */
    uint32_t route_slot_bytes[kMaxRanks];      /* per-rank routing-list capacity. */
    uint32_t activation_slot_bytes[kMaxRanks]; /* per-rank BF16 reference + exact E4M3 q/F32 scale staging. */
    uint32_t result_slot_bytes[kMaxRanks * kMaxRanks]; /* [producer rank][output rank]. */
    uint32_t owned_count;            /* number of set (layer, expert) ownership bits, all helpers. */
    uint32_t owner_slot_bytes;       /* fixed bitmap bytes at owner_offset. */
    uint64_t owned_bytes;            /* resident expert bytes charged against helper budget. */
    uint64_t owner_offset;           /* [kMaxRanks helpers][kMaxLayers][kExperts] bitsets. */
    uint32_t layer_map_slot_bytes;   /* int32_t[kMaxLayers][kExperts], -1 means protected/not regular. */
    uint32_t layer_map_ready[kMaxLayers];    /* engine release-stores after publishing one layer map. */
    uint32_t layer_map_consumed[kMaxLayers]; /* helper release-stores after rebuilding that layer. */
    uint64_t layer_map_offset;
    uint32_t next_sequence;         /* atomic ticket allocator; sequence 0 is reserved. */
    uint32_t publish_sequence;      /* newest published ticket; helper FUTEX_WAITs here. */
    LaneSignal lane[kLanes];          /* GPU-only request/done words, one cache line per lane (lane_of). */
    uint32_t small_class_count;     /* helper-set: enabled small-lane sizes, 1..kSmallClasses. */
    uint32_t small_class_rows[kSmallClasses]; /* helper-set: strictly ascending row limits; a step above the last goes to the large lane. */
    /* Helpers serving this segment (helper 0 sets it). 1: one helper serves every rank's slot. 2: a one-rank
     * engine (the R9700) with two helpers (the XTX cards); helper h owns bitmap h and serves rank slot h, and
     * the engine sends each route to the slot of the helper that owns its expert. */
    uint32_t helper_count;
    /* Set by a helper whose lanes stopped (error or shutdown): an engine still waiting traps at once
     * instead of spinning on a done word that will never move. */
    uint32_t helper_fault;
    uint64_t ring_offset;           /* Request[kRingSlots], page aligned. */
    uint64_t route_offset[kMaxRanks];
    uint64_t activation_offset[kMaxRanks];
    uint64_t result_offset[kMaxRanks * kMaxRanks]; /* [producer rank][output rank], like result_slot_bytes. */
    uint64_t segment_bytes;         /* exact mmap/ftruncate/hipHostRegister-able length. */
    uint32_t intermediate;          /* expert intermediate width (RCS09-03): part of the shape both sides must agree on. */
    uint32_t activation_group;      /* activation quantisation group width. */
    uint64_t shape_hash;            /* shape_hash() of the shape fields: a fast agreement check, never the only one. */
    uint64_t reserved[14];
};
static_assert(sizeof(Header) <= kPage);

constexpr uint64_t align_page(uint64_t n) { return (n + kPage - 1) & ~(uint64_t(kPage) - 1); }
constexpr uint32_t route_slot_bytes() { return kMaxRoutes * sizeof(Route); }
constexpr uint32_t activation_slot_bytes() { return kActivationBf16Bytes + kActivationQBytes + kActivationScaleBytes; }
constexpr uint32_t result_slot_bytes() { return kMaxRows * kHidden * sizeof(uint16_t); }
constexpr uint32_t owner_slot_bytes() { return kMaxLayers * ((kExperts + 7u) / 8u); }
constexpr uint32_t layer_map_slot_bytes() { return kMaxLayers * kExperts * sizeof(int32_t); }

struct Layout {
    uint64_t ring;
    uint64_t owner;
    uint64_t layer_map;
    uint64_t route[kMaxRanks];
    uint64_t activation[kMaxRanks];
    uint64_t result[kMaxRanks * kMaxRanks];
    uint64_t bytes;
    uint32_t route_bytes, activation_bytes, result_bytes, owner_bytes, layer_map_bytes;   /* per-slot sizes the header records */
};

constexpr Layout layout() {
    Layout x{};
    uint64_t at = kPage;
    x.ring = at;
    at = align_page(at + sizeof(Request) * kRingSlots);
    x.owner = at;
    at = align_page(at + (uint64_t)kMaxRanks * owner_slot_bytes());
    x.layer_map = at;
    at = align_page(at + layer_map_slot_bytes());
    for (uint32_t r = 0; r < kMaxRanks; ++r) {
        x.route[r] = at; at = align_page(at + route_slot_bytes());
        x.activation[r] = at; at = align_page(at + activation_slot_bytes());
        for (uint32_t o = 0; o < kMaxRanks; ++o) {
            x.result[r * kMaxRanks + o] = at;
            at = align_page(at + result_slot_bytes());
        }
    }
    x.bytes = align_page(at);
    x.route_bytes = route_slot_bytes(); x.activation_bytes = activation_slot_bytes(); x.result_bytes = result_slot_bytes();
    x.owner_bytes = owner_slot_bytes(); x.layer_map_bytes = layer_map_slot_bytes();
    return x;
}

/* The same layout from a runtime shape (RCS09-02): every region is sized from the model's hidden width, top-k, expert count, row
 * capacity and activation quantisation group, with checked 64-bit arithmetic, instead of the Flash-Next constants above. The
 * protocol bounds (kMaxRanks, kMaxLayers, the lane ring) stay compile-time. Order and page alignment are exactly layout()'s, so
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

inline bool layout_for(const LayoutShape& s, Layout* out) {
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
    if (!out || !s.hidden || !s.top_k || !s.experts || !s.max_rows || !s.act_group || !s.intermediate || s.top_k > s.experts) return false;
    uint64_t routes = 0, rows_hidden = 0, groups = 0, scale_cols = 0, owner_row = 0;
    uint64_t route_b = 0, bf16_b = 0, q_b = 0, scale_b = 0, act_b = 0, result_b = 0, owner_b = 0, map_b = 0, tmp = 0;
    if (!mul(s.max_rows, s.top_k, &routes) || !mul(routes, sizeof(Route), &route_b)) return false;
    if (!mul(s.max_rows, s.hidden, &rows_hidden) || !mul(rows_hidden, sizeof(uint16_t), &bf16_b)) return false;
    q_b = rows_hidden;
    groups = (uint64_t)s.hidden / s.act_group;   /* the quantiser works in whole groups: a tail is not defined, so refuse it */
    if (!groups || groups * s.act_group != s.hidden) return false;
    scale_cols = groups;
    if (!mul(s.max_rows, scale_cols, &tmp) || !mul(tmp, sizeof(float), &scale_b)) return false;
    if (!add(bf16_b, q_b, &tmp) || !add(tmp, scale_b, &act_b)) return false;
    result_b = bf16_b;
    owner_row = ((uint64_t)s.experts + 7u) / 8u;
    if (!mul(kMaxLayers, owner_row, &owner_b)) return false;
    if (!mul((uint64_t)kMaxLayers * s.experts, sizeof(int32_t), &map_b)) return false;
    for (uint64_t v : {route_b, act_b, result_b, owner_b, map_b})
        if (v > UINT32_MAX) return false;   /* the header records each slot size in 32 bits */
    Layout x{};
    uint64_t at = kPage;
    x.ring = at;
    if (!add(at, (uint64_t)sizeof(Request) * kRingSlots, &tmp) || !page(tmp, &at)) return false;
    x.owner = at;
    if (!mul(kMaxRanks, owner_b, &tmp) || !add(at, tmp, &tmp) || !page(tmp, &at)) return false;
    x.layer_map = at;
    if (!add(at, map_b, &tmp) || !page(tmp, &at)) return false;
    for (uint32_t r = 0; r < kMaxRanks; ++r) {
        x.route[r] = at;
        if (!add(at, route_b, &tmp) || !page(tmp, &at)) return false;
        x.activation[r] = at;
        if (!add(at, act_b, &tmp) || !page(tmp, &at)) return false;
        for (uint32_t o = 0; o < kMaxRanks; ++o) {
            x.result[r * kMaxRanks + o] = at;
            if (!add(at, result_b, &tmp) || !page(tmp, &at)) return false;
        }
    }
    if (!page(at, &x.bytes)) return false;
    x.route_bytes = (uint32_t)route_b; x.activation_bytes = (uint32_t)act_b; x.result_bytes = (uint32_t)result_b;
    x.owner_bytes = (uint32_t)owner_b; x.layer_map_bytes = (uint32_t)map_b;
    *out = x;
    return true;
}

/* Lay out and stamp a segment for this shape. The caller has mapped at least layout_for(shape).bytes. False (and nothing written)
 * when the shape does not lay out. */
inline bool init(void* base, uint32_t layers, const LayoutShape& shape) {
    Layout l{};
    if (!base || !layers || layers > kMaxLayers || !layout_for(shape, &l)) return false;
    std::memset(base, 0, static_cast<size_t>(l.bytes));
    auto* h = static_cast<Header*>(base);
    h->magic = kMagic; h->version = kVersion; h->header_bytes = sizeof(Header);
    h->layer_count = layers; h->rank_count = kMaxRanks; h->ring_slots = kRingSlots;
    h->max_rows = shape.max_rows; h->hidden = shape.hidden; h->top_k = shape.top_k; h->expert_count = shape.experts;
    h->intermediate = shape.intermediate; h->activation_group = shape.act_group; h->shape_hash = shape_hash(shape);
    h->owned_count = 0; h->owner_slot_bytes = l.owner_bytes; h->owned_bytes = 0;
    h->owner_offset = l.owner;
    h->layer_map_slot_bytes = l.layer_map_bytes; h->layer_map_offset = l.layer_map;
    std::memset(static_cast<uint8_t*>(base) + l.layer_map, 0xff, l.layer_map_bytes);
    h->next_sequence = 1; h->ring_offset = l.ring; h->segment_bytes = l.bytes;
    h->small_class_count = 1; h->small_class_rows[0] = 8;   /* the helper overwrites these before it publishes helper_count */
    for (uint32_t r = 0; r < kMaxRanks; ++r) {
        h->route_slot_bytes[r] = l.route_bytes;
        h->activation_slot_bytes[r] = l.activation_bytes;
        h->route_offset[r] = l.route[r]; h->activation_offset[r] = l.activation[r];
        for (uint32_t o = 0; o < kMaxRanks; ++o) {
            const uint32_t i = r * kMaxRanks + o;
            h->result_slot_bytes[i] = l.result_bytes;
            h->result_offset[i] = l.result[i];
        }
    }
    return true;
}

/* A mapped segment is acceptable when it is a v14 segment of exactly the shape this side expects, and its layout is the one that
 * shape implies: every offset, slot size and length is reconstructed from the header's own shape fields, so a corrupt or foreign
 * header cannot point a request outside the mapping. A different shape, a stale protocol version or a malformed layout is refused
 * before any request is published or any expert buffer is touched. */
inline bool valid(const Header* h, size_t mapped, const LayoutShape& expect) {
    if (!h || h->magic != kMagic || h->version != kVersion || h->header_bytes != sizeof(Header) ||
        h->rank_count != kMaxRanks || h->ring_slots != kRingSlots ||
        h->layer_count == 0 || h->layer_count > kMaxLayers ||
        h->small_class_count == 0 || h->small_class_count > kSmallClasses) return false;
    const LayoutShape got{h->hidden, h->top_k, h->expert_count, h->max_rows, h->activation_group, h->intermediate};
    if (got != expect || h->shape_hash != shape_hash(got)) return false;
    Layout l{};
    if (!layout_for(got, &l)) return false;
    for (uint32_t c = 0; c < h->small_class_count; ++c)
        if (h->small_class_rows[c] == 0 || h->small_class_rows[c] > h->max_rows ||
            (c && h->small_class_rows[c] <= h->small_class_rows[c - 1])) return false;
    if (h->ring_offset != l.ring || h->owner_offset != l.owner ||
        h->owner_slot_bytes != l.owner_bytes ||
        h->layer_map_offset != l.layer_map || h->layer_map_slot_bytes != l.layer_map_bytes ||
        h->segment_bytes != l.bytes || mapped < l.bytes) return false;
    for (uint32_t r = 0; r < kMaxRanks; ++r)
        /* Two helpers share slot 0's activation staging (helper 0 aliases the others to it). */
        if (h->route_offset[r] != l.route[r] ||
            (h->activation_offset[r] != l.activation[r] && h->activation_offset[r] != l.activation[0]) ||
            h->route_slot_bytes[r] != l.route_bytes ||
            h->activation_slot_bytes[r] != l.activation_bytes) return false;
    for (uint32_t i = 0; i < kMaxRanks * kMaxRanks; ++i)
        if (h->result_offset[i] != l.result[i] ||
            h->result_slot_bytes[i] != l.result_bytes) return false;
    return true;
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

inline uint8_t* owner_bits(void* base, uint32_t helper) {
    auto* h = static_cast<Header*>(base);
    return ptr<uint8_t>(base, h->owner_offset) + (size_t)helper * h->owner_slot_bytes;
}
inline const uint8_t* owner_bits(const void* base, uint32_t helper) {
    auto* h = static_cast<const Header*>(base);
    return ptr<uint8_t>(base, h->owner_offset) + (size_t)helper * h->owner_slot_bytes;
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
/* Bit index of (layer, expert) in an ownership bitmap: the stride is the segment's expert count (Header::expert_count). */
inline uint64_t owner_bit(uint32_t layer, uint32_t expert, uint32_t experts) {
    return (uint64_t)layer * experts + expert;
}
inline bool owns(const void* base, uint32_t layer, uint32_t expert, uint32_t helper) {
    const auto* h = static_cast<const Header*>(base);
    if (!h || layer >= h->layer_count || expert >= h->expert_count || helper >= kMaxRanks) return false;
    const uint64_t bit = owner_bit(layer, expert, h->expert_count);
    return (owner_bits(base, helper)[bit >> 3] & (uint8_t)(1u << (bit & 7u))) != 0;
}
inline bool set_owned(void* base, uint32_t layer, uint32_t expert, uint32_t helper) {
    auto* h = static_cast<Header*>(base);
    if (!h || layer >= h->layer_count || expert >= h->expert_count || helper >= kMaxRanks) return false;
    const uint64_t bit = owner_bit(layer, expert, h->expert_count);
    uint8_t* b = owner_bits(base, helper) + (bit >> 3);
    const uint8_t mask = (uint8_t)(1u << (bit & 7u));
    if (*b & mask) return false;
    *b |= mask;
    __atomic_fetch_add(&h->owned_count, 1u, __ATOMIC_ACQ_REL);
    return true;
}
inline bool clear_owned(void* base, uint32_t layer, uint32_t expert, uint32_t helper) {
    auto* h = static_cast<Header*>(base);
    if (!h || layer >= h->layer_count || expert >= h->expert_count || helper >= kMaxRanks) return false;
    const uint64_t bit = owner_bit(layer, expert, h->expert_count);
    uint8_t* b = owner_bits(base, helper) + (bit >> 3);
    const uint8_t mask = (uint8_t)(1u << (bit & 7u));
    if (!(*b & mask)) return false;
    *b &= (uint8_t)~mask;
    __atomic_fetch_sub(&h->owned_count, 1u, __ATOMIC_ACQ_REL);
    return true;
}

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
inline void wait_until(uint32_t* p, uint32_t value) {
    for (;;) {
        const uint32_t seen = load(p);
        if (seen == value) return;
        if (futex_wait(p, seen) < 0 && errno != EAGAIN && errno != EINTR) return;
    }
}

} // namespace r3::moe_shm
