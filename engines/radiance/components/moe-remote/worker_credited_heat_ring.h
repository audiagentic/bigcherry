#pragma once
// Inactive RHP02 B2 SPSC transport. The caller owns the page-aligned span;
// the sole producer never allocates, waits, acquires locks or touches a GPU.
// The span is formatted exclusively before either endpoint attaches.
#include "worker_credited_heat_source.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace r3::policy::heat_ring {
constexpr size_t kPage = 4096;  // transport format page
constexpr size_t kLine = 64;    // separate cache lines for cursor publication
constexpr uint64_t kMagic = UINT64_C(0x5248503032484541);
constexpr uint32_t kVersion = 1;
static_assert(__atomic_always_lock_free(8, nullptr), "requires lock-free interprocess 64-bit atomics");

struct Shape { uint32_t rank, layers, experts; uint64_t capacity; };
enum class Error { NONE, POINTER, ALIGN, SHAPE, SIZE, MAGIC, VERSION, MISMATCH, CURSOR, ATOMIC };
enum class Write { STORED, FULL, INVALID };
struct alignas(kLine) AtomicLine { uint64_t value; unsigned char pad[kLine-8]; };
static_assert(sizeof(AtomicLine) == kLine);
struct alignas(kLine) HeaderData {
    uint64_t magic, span_bytes, row_bytes, capacity;
    uint32_t version, header_bytes, rank, layers, experts, reserved;
    AtomicLine produced, consumed, generation, dropped, generation_drop_base, gap_number, last_gap_credit;
};
static_assert(sizeof(HeaderData) <= kPage);
struct alignas(kPage) Header : HeaderData { unsigned char padding[kPage - sizeof(HeaderData)]; };
static_assert(sizeof(Header) == kPage);
struct alignas(kLine) Slot {
    uint64_t commit, generation, run_generation, owner_epoch, source_step, bank_generation;
    uint64_t credit_sequence, issued_watermark, completed_watermark, credited_watermark;
    uint64_t dropped_watermark, observed_monotonic_ns, weighted_sum, observation_count;
    uint32_t rank, layer, reports, phase, source_complete, reserved;
    // int32_t counts[experts] follows inline.
};
static_assert(sizeof(Slot) % kLine == 0);
inline uint64_t get(const uint64_t* v, int order = __ATOMIC_ACQUIRE) {
    return __atomic_load_n(v, order);
}
inline void put(uint64_t* v, uint64_t n, int order = __ATOMIC_RELEASE) {
    __atomic_store_n(v, n, order);
}
inline bool layout(Shape s, size_t* stride, size_t* bytes) {
    if (!stride || !bytes || !s.layers || !s.experts || !s.capacity ||
        s.experts > (SIZE_MAX - sizeof(Slot) - kLine + 1) / sizeof(int32_t))
        return false;
    size_t n = sizeof(Slot) + size_t(s.experts) * sizeof(int32_t);
    *stride = (n + kLine - 1) & ~(kLine - 1);
    if (s.capacity > (SIZE_MAX - kPage) / *stride) return false;
    *bytes = kPage + size_t(s.capacity) * *stride;
    return true;
}
inline bool aligned(const void* p) {
    return p && reinterpret_cast<uintptr_t>(p) % kPage == 0;
}
inline bool format(void* base, size_t size, Shape s, Error* err = nullptr) {
    auto fail = [err](Error e) { if (err) *err = e; return false; };
    if (!base) return fail(Error::POINTER);
    if (!aligned(base)) return fail(Error::ALIGN);
    size_t stride, bytes;
    if (!layout(s, &stride, &bytes)) return fail(Error::SHAPE);
    if (size < bytes) return fail(Error::SIZE);
    std::memset(base, 0, bytes);
    auto* h = reinterpret_cast<Header*>(base);
    h->magic = kMagic; h->version = kVersion; h->header_bytes = kPage;
    h->span_bytes = bytes; h->row_bytes = stride; h->capacity = s.capacity;
    h->rank = s.rank; h->layers = s.layers; h->experts = s.experts;
    if (err) *err = Error::NONE;
    return true;
}
inline Header* validate(void* base, size_t size, Shape s, Error* err = nullptr) {
    auto fail = [err](Error e) -> Header* { if (err) *err = e; return nullptr; };
    if (!base) return fail(Error::POINTER);
    if (!aligned(base)) return fail(Error::ALIGN);
    if (size < kPage) return fail(Error::SIZE);
    size_t stride, bytes;
    if (!layout(s, &stride, &bytes)) return fail(Error::SHAPE);
    auto* h = reinterpret_cast<Header*>(base);
    if (h->magic != kMagic || h->header_bytes != kPage ||
        h->span_bytes != bytes || h->row_bytes != stride) return fail(Error::MAGIC);
    if (h->version != kVersion) return fail(Error::VERSION);
    if (h->rank != s.rank || h->layers != s.layers ||
        h->experts != s.experts || h->capacity != s.capacity) return fail(Error::MISMATCH);
    if (size < bytes) return fail(Error::SIZE);
    if (!__atomic_is_lock_free(8, &h->produced.value) ||
        !__atomic_is_lock_free(8, &h->consumed.value)) return fail(Error::ATOMIC);
    uint64_t p = get(&h->produced.value), c = get(&h->consumed.value);
    if (p < c || p-c > s.capacity || p == UINT64_MAX || c == UINT64_MAX)
        return fail(Error::CURSOR);
    if (err) *err = Error::NONE;
    return h;
}
class Producer {
    Header* h_ = nullptr;
    unsigned char* base_ = nullptr;
    Shape shape_{};
public:
    bool attach(void* base, size_t size, Shape shape, Error* err = nullptr) {
        h_ = nullptr;
        auto* h = validate(base, size, shape, err);
        if (!h) return false;
        h_ = h; base_ = static_cast<unsigned char*>(base); shape_ = shape;
        return true;
    }
    // One producer owns generation changes; no concurrent format/reset.
    bool start(uint64_t generation) {
        if (!h_ || !generation || generation <= get(&h_->generation.value)) return false;
        put(&h_->generation_drop_base.value, get(&h_->dropped.value));
        put(&h_->generation.value, generation);
        return true;
    }
    Write push(const CreditedHeatRow& r) {
        if (!h_ || !get(&h_->generation.value) ||
            r.rank != shape_.rank || r.layer >= shape_.layers ||
            r.counts.size() != shape_.experts || !r.credit_sequence) return Write::INVALID;
        uint64_t p = get(&h_->produced.value, __ATOMIC_RELAXED);
        uint64_t c = get(&h_->consumed.value);
        if (p < c || p-c > shape_.capacity || p == UINT64_MAX) return Write::INVALID;
        if (p-c == shape_.capacity) {
            uint64_t drops = get(&h_->dropped.value, __ATOMIC_RELAXED);
            uint64_t gaps = get(&h_->gap_number.value, __ATOMIC_RELAXED);
            if (drops == UINT64_MAX || gaps == UINT64_MAX) return Write::INVALID;
            put(&h_->last_gap_credit.value, r.credit_sequence);
            put(&h_->gap_number.value, gaps+1);
            put(&h_->dropped.value, drops+1);
            return Write::FULL;
        }
        uint64_t total_dropped = get(&h_->dropped.value, __ATOMIC_RELAXED);
        uint64_t base_dropped = get(&h_->generation_drop_base.value, __ATOMIC_RELAXED);
        if (total_dropped < base_dropped ||
            r.dropped_watermark > UINT64_MAX - (total_dropped - base_dropped)) return Write::INVALID;
        auto* slot = reinterpret_cast<Slot*>(base_ + kPage + size_t(p % shape_.capacity) * h_->row_bytes);
        slot->generation = get(&h_->generation.value, __ATOMIC_RELAXED);
        slot->run_generation=r.run_generation; slot->owner_epoch=r.owner_epoch;
        slot->source_step=r.source_step; slot->bank_generation=r.bank_generation;
        slot->credit_sequence=r.credit_sequence; slot->issued_watermark=r.issued_watermark;
        slot->completed_watermark=r.completed_watermark;
        slot->credited_watermark=r.credited_watermark;
        slot->dropped_watermark=r.dropped_watermark+(total_dropped-base_dropped);
        slot->observed_monotonic_ns=r.observed_monotonic_ns;
        slot->weighted_sum=r.weighted_sum; slot->observation_count=r.observation_count;
        slot->rank=r.rank; slot->layer=r.layer; slot->reports=r.reports;
        slot->phase=static_cast<uint32_t>(r.phase); slot->source_complete=r.source_complete;
        std::memcpy(reinterpret_cast<unsigned char*>(slot)+sizeof(Slot),
                    r.counts.data(), size_t(shape_.experts)*sizeof(int32_t));
        put(&slot->commit, p+1);       // mutation: dropping release must fail concurrent test
        put(&h_->produced.value, p+1);
        return Write::STORED;
    }
    uint64_t drops() const { return h_ ? get(&h_->dropped.value) : 0; }
    uint64_t gap_number() const { return h_ ? get(&h_->gap_number.value) : 0; }
    uint64_t last_gap_credit() const { return h_ ? get(&h_->last_gap_credit.value) : 0; }
};
class Consumer final : public CreditedRowReader {
    Header* h_ = nullptr;
    unsigned char* base_ = nullptr;
    Shape shape_{};
    uint64_t generation_ = 0, dropped_ = 0;
public:
    bool attach(void* base, size_t size, Shape shape, Error* err = nullptr) {
        h_ = nullptr;
        auto* h = validate(base, size, shape, err);
        if (!h) return false;
        h_ = h; base_ = static_cast<unsigned char*>(base); shape_ = shape;
        generation_ = 0; dropped_ = 0;
        return true;
    }
    uint64_t generation() const { return h_ ? get(&h_->generation.value) : 0; }
    CreditedRead read_batch(std::vector<CreditedHeatRow>* out) override {
        if (!h_ || !out) return CreditedRead::INVALID;
        out->clear();
        uint64_t gen = get(&h_->generation.value);
        if (!gen) return CreditedRead::EMPTY; // late engine
        if (generation_ && gen != generation_) {
            generation_ = gen;
            dropped_ = 0;
            return CreditedRead::GAP; // restart never treated as current baseline
        }
        if (!generation_) generation_ = gen;
        uint64_t base_drops = get(&h_->generation_drop_base.value);
        uint64_t total_drops = get(&h_->dropped.value);
        if (total_drops < base_drops) return CreditedRead::INVALID;
        uint64_t drops = total_drops - base_drops;
        if (drops != dropped_) {
            dropped_ = drops;
            return CreditedRead::GAP;
        }
        uint64_t c = get(&h_->consumed.value, __ATOMIC_RELAXED);
        uint64_t p = get(&h_->produced.value);
        if (p < c || p-c > shape_.capacity || p == UINT64_MAX) return CreditedRead::INVALID;
        if (p == c) return CreditedRead::EMPTY;
        for (; c < p; ++c) {
            auto* slot = reinterpret_cast<Slot*>(base_ + kPage + size_t(c % shape_.capacity)*h_->row_bytes);
            if (get(&slot->commit) != c+1) { out->clear(); return CreditedRead::INVALID; }
            if (slot->generation != gen) {
                out->clear();
                put(&h_->consumed.value, c+1);
                return CreditedRead::GAP;
            }
            CreditedHeatRow r;
            r.run_generation=slot->run_generation; r.owner_epoch=slot->owner_epoch;
            r.source_step=slot->source_step; r.bank_generation=slot->bank_generation;
            r.credit_sequence=slot->credit_sequence; r.issued_watermark=slot->issued_watermark;
            r.completed_watermark=slot->completed_watermark;
            r.credited_watermark=slot->credited_watermark;
            r.dropped_watermark=slot->dropped_watermark;
            r.observed_monotonic_ns=slot->observed_monotonic_ns;
            r.weighted_sum=slot->weighted_sum; r.observation_count=slot->observation_count;
            r.rank=slot->rank; r.layer=slot->layer; r.reports=slot->reports;
            r.phase=static_cast<CreditedPhase>(slot->phase);
            r.source_complete=slot->source_complete == 1;
            const auto* counts = reinterpret_cast<const int32_t*>(
                reinterpret_cast<const unsigned char*>(slot)+sizeof(Slot));
            r.counts.assign(counts, counts+shape_.experts);
            if (get(&h_->generation.value) != gen) { out->clear(); return CreditedRead::GAP; }
            out->push_back(std::move(r));
            put(&h_->consumed.value, c+1); // release slot only after copying
        }
        if (get(&h_->generation.value) != gen || get(&h_->dropped.value) - get(&h_->generation_drop_base.value) != drops) {
            out->clear(); return CreditedRead::GAP;
        }
        return CreditedRead::ROWS;
    }
};
} // namespace r3::policy::heat_ring
