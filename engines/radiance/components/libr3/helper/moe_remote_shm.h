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
constexpr uint32_t kVersion = 1;
constexpr uint32_t kPage = 4096;
constexpr uint32_t kMaxRanks = 2;       /* Flash-Next configuration of record: TP=2. */
constexpr uint32_t kRingSlots = 16;     /* metadata only; each rank owns one bulk-data slot. */
constexpr uint32_t kHidden = 2560;
constexpr uint32_t kTopK = 10;
constexpr uint32_t kExperts = 512;
constexpr uint32_t kMaxRows = 4096;
constexpr uint32_t kMaxRoutes = kMaxRows * kTopK;

struct Route {
    int32_t token;       /* activation/result row to which this expert contribution belongs. */
    int32_t expert;      /* global expert id; addresses the .rad expert group directly. */
    uint16_t weight_bf16;/* router weight exactly as supplied by the patch ABI. */
    uint16_t reserved;
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
    uint32_t reserved0;
    uint64_t route_offset;         /* routed (token,expert,weight) list for this rank. */
    uint64_t activation_offset;    /* staged BF16 input rows for this rank. */
    uint32_t activation_stride;    /* bytes between staged rows; allows ABI stride to differ. */
    uint32_t result_stride[kMaxRanks]; /* BF16 row stride for each TP rank's result slot. */
    uint64_t result_offset[kMaxRanks]; /* partial-output slots; helper writes only req.rank. */
    uint64_t reserved[4];
};
static_assert(sizeof(Request) % 64 == 0);

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
    uint32_t activation_slot_bytes[kMaxRanks]; /* per-rank staged BF16 activation capacity. */
    uint32_t result_slot_bytes[kMaxRanks];     /* per-rank BF16 partial capacity. */
    uint32_t next_sequence;         /* atomic ticket allocator; sequence 0 is reserved. */
    uint32_t publish_sequence;      /* newest published ticket; helper FUTEX_WAITs here. */
    uint64_t ring_offset;           /* Request[kRingSlots], page aligned. */
    uint64_t route_offset[kMaxRanks];
    uint64_t activation_offset[kMaxRanks];
    uint64_t result_offset[kMaxRanks];
    uint64_t segment_bytes;         /* exact mmap/ftruncate/hipHostRegister-able length. */
    uint64_t reserved[16];
};
static_assert(sizeof(Header) <= kPage);

constexpr uint64_t align_page(uint64_t n) { return (n + kPage - 1) & ~(uint64_t(kPage) - 1); }
constexpr uint32_t route_slot_bytes() { return kMaxRoutes * sizeof(Route); }
constexpr uint32_t activation_slot_bytes() { return kMaxRows * kHidden * sizeof(uint16_t); }
constexpr uint32_t result_slot_bytes() { return kMaxRows * kHidden * sizeof(uint16_t); }

struct Layout {
    uint64_t ring;
    uint64_t route[kMaxRanks];
    uint64_t activation[kMaxRanks];
    uint64_t result[kMaxRanks];
    uint64_t bytes;
};

constexpr Layout layout() {
    Layout x{};
    uint64_t at = kPage;
    x.ring = at;
    at = align_page(at + sizeof(Request) * kRingSlots);
    for (uint32_t r = 0; r < kMaxRanks; ++r) {
        x.route[r] = at; at = align_page(at + route_slot_bytes());
        x.activation[r] = at; at = align_page(at + activation_slot_bytes());
        x.result[r] = at; at = align_page(at + result_slot_bytes());
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
    h->next_sequence = 1; h->ring_offset = l.ring; h->segment_bytes = l.bytes;
    for (uint32_t r = 0; r < kMaxRanks; ++r) {
        h->route_slot_bytes[r] = route_slot_bytes();
        h->activation_slot_bytes[r] = activation_slot_bytes();
        h->result_slot_bytes[r] = result_slot_bytes();
        h->route_offset[r] = l.route[r]; h->activation_offset[r] = l.activation[r];
        h->result_offset[r] = l.result[r];
    }
}

inline bool valid(const Header* h, size_t mapped) {
    if (!h || h->magic != kMagic || h->version != kVersion || h->header_bytes != sizeof(Header) ||
        h->rank_count != kMaxRanks || h->ring_slots != kRingSlots || h->max_rows != kMaxRows ||
        h->hidden != kHidden || h->top_k != kTopK || h->expert_count != kExperts) return false;
    const Layout l = layout();
    if (h->ring_offset != l.ring || h->segment_bytes != l.bytes || mapped < l.bytes) return false;
    for (uint32_t r = 0; r < kMaxRanks; ++r)
        if (h->route_offset[r] != l.route[r] || h->activation_offset[r] != l.activation[r] ||
            h->result_offset[r] != l.result[r] || h->route_slot_bytes[r] != route_slot_bytes() ||
            h->activation_slot_bytes[r] != activation_slot_bytes() ||
            h->result_slot_bytes[r] != result_slot_bytes()) return false;
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
