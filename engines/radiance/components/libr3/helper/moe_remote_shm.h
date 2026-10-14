#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace r3::moe_shm {

constexpr uint64_t kMagic = 0x31454f4d33485253ull; /* "SRH3MOE1": reject unrelated/stale shm. */
constexpr uint32_t kVersion = 9;   /* 9: small (pre-queued) and large (sized on arrival) lanes a rank */
constexpr uint32_t kPage = 4096;
constexpr uint32_t kMaxRanks = 2;       /* Flash-Next configuration of record: TP=2. */
constexpr uint32_t kMaxLayers = 128;     /* ownership bitmap bound; Flash-Next is below this. */
constexpr uint32_t kRingSlots = 16;
/* Two lanes a producer rank: lane `rank` takes steps of at most Header::small_rows rows and the helper
 * pre-queues its GPU work at that fixed size (no host in the loop); lane `rank + kMaxRanks` takes the larger
 * (prefill) steps, which the helper launches sized to the request when it arrives. A rank publishes one
 * request at a time, so both lanes share that rank's route/activation/result slots. */
constexpr uint32_t kLanes = 2 * kMaxRanks;
constexpr uint32_t kLaneSlots = kRingSlots / kLanes; /* lane-private metadata ring; bulk slots stay per-rank. */
constexpr uint32_t lane_of(uint32_t rank, uint32_t large) { return rank + (large ? kMaxRanks : 0u); }
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
    uint64_t reserved[2];
};
static_assert(sizeof(Request) % 64 == 0);

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
    uint32_t owned_count;            /* number of set (layer, expert) ownership bits. */
    uint32_t owner_slot_bytes;       /* fixed bitmap bytes at owner_offset. */
    uint64_t owned_bytes;            /* resident expert bytes charged against helper budget. */
    uint64_t owner_offset;           /* [kMaxLayers][kExperts] bitset. */
    uint32_t layer_map_slot_bytes;   /* int32_t[kMaxLayers][kExperts], -1 means protected/not regular. */
    uint32_t layer_map_ready[kMaxLayers];    /* engine release-stores after publishing one layer map. */
    uint32_t layer_map_consumed[kMaxLayers]; /* helper release-stores after rebuilding that layer. */
    uint64_t layer_map_offset;
    uint32_t next_sequence;         /* atomic ticket allocator; sequence 0 is reserved. */
    uint32_t publish_sequence;      /* newest published ticket; helper FUTEX_WAITs here. */
    LaneSignal lane[kLanes];          /* GPU-only request/done words, one cache line per lane (lane_of). */
    uint32_t small_rows;            /* helper-set: steps above this many rows go to the large lane. */
    uint64_t ring_offset;           /* Request[kRingSlots], page aligned. */
    uint64_t route_offset[kMaxRanks];
    uint64_t activation_offset[kMaxRanks];
    uint64_t result_offset[kMaxRanks * kMaxRanks]; /* [producer rank][output rank], like result_slot_bytes. */
    uint64_t segment_bytes;         /* exact mmap/ftruncate/hipHostRegister-able length. */
    uint64_t reserved[16];
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
};

constexpr Layout layout() {
    Layout x{};
    uint64_t at = kPage;
    x.ring = at;
    at = align_page(at + sizeof(Request) * kRingSlots);
    x.owner = at;
    at = align_page(at + owner_slot_bytes());
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
    return x;
}

inline void init(void* base, uint32_t layers) {
    const Layout l = layout();
    std::memset(base, 0, static_cast<size_t>(l.bytes));
    auto* h = static_cast<Header*>(base);
    h->magic = kMagic; h->version = kVersion; h->header_bytes = sizeof(Header);
    h->layer_count = layers; h->rank_count = kMaxRanks; h->ring_slots = kRingSlots;
    h->max_rows = kMaxRows; h->hidden = kHidden; h->top_k = kTopK; h->expert_count = kExperts;
    h->owned_count = 0; h->owner_slot_bytes = owner_slot_bytes(); h->owned_bytes = 0;
    h->owner_offset = l.owner;
    h->layer_map_slot_bytes = layer_map_slot_bytes(); h->layer_map_offset = l.layer_map;
    std::memset(static_cast<uint8_t*>(base) + l.layer_map, 0xff, layer_map_slot_bytes());
    h->next_sequence = 1; h->ring_offset = l.ring; h->segment_bytes = l.bytes;
    for (uint32_t r = 0; r < kMaxRanks; ++r) {
        h->route_slot_bytes[r] = route_slot_bytes();
        h->activation_slot_bytes[r] = activation_slot_bytes();
        h->route_offset[r] = l.route[r]; h->activation_offset[r] = l.activation[r];
        for (uint32_t o = 0; o < kMaxRanks; ++o) {
            const uint32_t i = r * kMaxRanks + o;
            h->result_slot_bytes[i] = result_slot_bytes();
            h->result_offset[i] = l.result[i];
        }
    }
}

inline bool valid(const Header* h, size_t mapped) {
    if (!h || h->magic != kMagic || h->version != kVersion || h->header_bytes != sizeof(Header) ||
        h->rank_count != kMaxRanks || h->ring_slots != kRingSlots || h->max_rows != kMaxRows ||
        h->hidden != kHidden || h->top_k != kTopK || h->expert_count != kExperts ||
        h->layer_count == 0 || h->layer_count > kMaxLayers) return false;
    const Layout l = layout();
    if (h->ring_offset != l.ring || h->owner_offset != l.owner ||
        h->owner_slot_bytes != owner_slot_bytes() ||
        h->layer_map_offset != l.layer_map || h->layer_map_slot_bytes != layer_map_slot_bytes() ||
        h->segment_bytes != l.bytes || mapped < l.bytes) return false;
    for (uint32_t r = 0; r < kMaxRanks; ++r)
        if (h->route_offset[r] != l.route[r] || h->activation_offset[r] != l.activation[r] ||
            h->route_slot_bytes[r] != route_slot_bytes() ||
            h->activation_slot_bytes[r] != activation_slot_bytes()) return false;
    for (uint32_t i = 0; i < kMaxRanks * kMaxRanks; ++i)
        if (h->result_offset[i] != l.result[i] ||
            h->result_slot_bytes[i] != result_slot_bytes()) return false;
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

inline uint8_t* owner_bits(void* base) {
    auto* h = static_cast<Header*>(base);
    return ptr<uint8_t>(base, h->owner_offset);
}
inline const uint8_t* owner_bits(const void* base) {
    auto* h = static_cast<const Header*>(base);
    return ptr<uint8_t>(base, h->owner_offset);
}
inline int32_t* layer_slots(void* base, uint32_t layer) {
    auto* h = static_cast<Header*>(base);
    if (!h || layer >= h->layer_count) return nullptr;
    return ptr<int32_t>(base, h->layer_map_offset) + (size_t)layer * kExperts;
}
inline const int32_t* layer_slots(const void* base, uint32_t layer) {
    auto* h = static_cast<const Header*>(base);
    if (!h || layer >= h->layer_count) return nullptr;
    return ptr<int32_t>(base, h->layer_map_offset) + (size_t)layer * kExperts;
}
inline int32_t layer_slot(const void* base, uint32_t layer, uint32_t expert) {
    const int32_t* p = layer_slots(base, layer);
    return p && expert < kExperts ? p[expert] : -1;
}
inline uint64_t owner_bit(uint32_t layer, uint32_t expert) {
    return (uint64_t)layer * kExperts + expert;
}
inline bool owns(const void* base, uint32_t layer, uint32_t expert) {
    const auto* h = static_cast<const Header*>(base);
    if (!h || layer >= h->layer_count || expert >= kExperts) return false;
    const uint64_t bit = owner_bit(layer, expert);
    return (owner_bits(base)[bit >> 3] & (uint8_t)(1u << (bit & 7u))) != 0;
}
inline bool set_owned(void* base, uint32_t layer, uint32_t expert) {
    auto* h = static_cast<Header*>(base);
    if (!h || layer >= h->layer_count || expert >= kExperts) return false;
    const uint64_t bit = owner_bit(layer, expert);
    uint8_t* b = owner_bits(base) + (bit >> 3);
    const uint8_t mask = (uint8_t)(1u << (bit & 7u));
    if (*b & mask) return false;
    *b |= mask;
    ++h->owned_count;
    return true;
}
inline bool clear_owned(void* base, uint32_t layer, uint32_t expert) {
    auto* h = static_cast<Header*>(base);
    if (!h || layer >= h->layer_count || expert >= kExperts) return false;
    const uint64_t bit = owner_bit(layer, expert);
    uint8_t* b = owner_bits(base) + (bit >> 3);
    const uint8_t mask = (uint8_t)(1u << (bit & 7u));
    if (!(*b & mask)) return false;
    *b &= (uint8_t)~mask;
    if (h->owned_count) --h->owned_count;
    return true;
}

inline uint32_t load(const uint32_t* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
inline void store(uint32_t* p, uint32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
inline uint32_t fetch_add(uint32_t* p, uint32_t v) { return __atomic_fetch_add(p, v, __ATOMIC_ACQ_REL); }
inline int futex_wait(uint32_t* p, uint32_t expected) {
    return static_cast<int>(::syscall(SYS_futex, p, FUTEX_WAIT, expected, nullptr, nullptr, 0));
}
inline int futex_wake(uint32_t* p, int n = 1) {
    return static_cast<int>(::syscall(SYS_futex, p, FUTEX_WAKE, n, nullptr, nullptr, 0));
}
inline void wait_until(uint32_t* p, uint32_t value) {
    for (;;) {
        const uint32_t seen = load(p);
        if (seen == value) return;
        if (futex_wait(p, seen) < 0 && errno != EAGAIN && errno != EINTR) return;
    }
}

} // namespace r3::moe_shm
