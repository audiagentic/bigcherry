#pragma once

// R3_MOE_WORKER_COMBINE (RAD18, RAD23): helper 0 adds the helpers' bf16 rows on the CPU so the one-rank engine reads one row
// a token over its link instead of two. The sum is written as fp16: the same two bytes as one bf16 row, but rounded 8x
// finer, since a bf16 sum measurably worsened KLD (RAD18 2026-10-23: p99 +0.275). A row with a value outside fp16 range
// is marked kCombineFallback in its first element and the engine reads that row from the two helper planes instead.
// Helper 0 combines on its large-lane thread chunk by chunk while the helpers' GPUs still write (RAD42), spinning with a
// yield on the chunk words only inside a combined step; the pool's workers sleep on futex words between jobs.
// Included only by r3_moe_worker.cpp, after worker_container.h (stopping()).

#include <linux/futex.h>
#include <sys/syscall.h>
#include <ctime>
#include "combine_f16.h"

namespace {


/* futex wait on a process-private word with a timeout, so a stop request is seen without polling the word. */
inline void combine_futex_wait(uint32_t* p, uint32_t expected, long timeout_ms, bool shared) {
    timespec ts{timeout_ms / 1000, (timeout_ms % 1000) * 1000000L};
    (void)::syscall(SYS_futex, p, shared ? FUTEX_WAIT : FUTEX_WAIT_PRIVATE, expected, timeout_ms > 0 ? &ts : nullptr,
                    nullptr, 0);
}
inline void combine_futex_wake(uint32_t* p, bool shared) {
    (void)::syscall(SYS_futex, p, shared ? FUTEX_WAKE : FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
}

/* The calling thread plus persistent workers; a job is split into token chunks taken from a counter. Workers sleep on
 * gen_ between jobs; the caller sleeps on busy_ until the last worker finishes. */
class CombinePool {
public:
    explicit CombinePool(uint32_t threads) {
        for (uint32_t i = 1; i < threads; ++i) workers_.emplace_back([this] { run(); });
    }
    ~CombinePool() {
        __atomic_store_n(&quit_, 1u, __ATOMIC_RELEASE);
        __atomic_fetch_add(&gen_, 1u, __ATOMIC_ACQ_REL);
        combine_futex_wake(&gen_, false);
        for (auto& t : workers_) t.join();
    }
    /* mask[t]: bit k set when helper k wrote row t. Rows with no bit are not read by the engine. */
    void combine(const uint16_t* const* src, uint32_t nsrc, uint16_t* out, const uint8_t* mask, uint32_t rows, uint32_t hidden) {
        nsrc_ = nsrc; for (uint32_t k = 0; k < nsrc; ++k) src_[k] = src[k];
        out_ = out; mask_ = mask; rows_ = rows; hidden_ = hidden;
        __atomic_store_n(&next_, 0u, __ATOMIC_RELAXED);
        __atomic_store_n(&busy_, (uint32_t)workers_.size(), __ATOMIC_RELAXED);
        __atomic_fetch_add(&gen_, 1u, __ATOMIC_RELEASE);
        combine_futex_wake(&gen_, false);
        work();
        for (uint32_t b; (b = __atomic_load_n(&busy_, __ATOMIC_ACQUIRE)) != 0;)
            combine_futex_wait(&busy_, b, 0, false);
    }

private:
    static constexpr uint32_t kChunk = 32;
    void work() {
        for (;;) {
            const uint32_t t0 = __atomic_fetch_add(&next_, kChunk, __ATOMIC_RELAXED);
            if (t0 >= rows_) return;
            const uint32_t t1 = std::min(rows_, t0 + kChunk);
            for (uint32_t t = t0; t < t1; ++t) {
                const uint8_t m = mask_[t];
                if (!m) continue;   /* no helper row: the engine does not read it */
                const uint64_t at = (uint64_t)t * hidden_;
                const uint16_t* rowp[r3::moe_shm::kCombineHelpers];
                for (uint32_t k = 0; k < nsrc_; ++k) rowp[k] = src_[k] + at;
                if (combine_row_f16(rowp, nsrc_, m, out_ + at, hidden_)) out_[at] = r3::moe_shm::kCombineFallback;
            }
        }
    }
    void run() {
        /* Start from the generation before any job (0, set before the workers exist), not from a load here: a worker that
         * starts after the first combine() has already bumped gen_ would otherwise take that job as seen and sleep, and
         * the caller would wait on busy_ forever (combine_pool_test). */
        uint32_t seen = 0;
        for (;;) {
            uint32_t g;
            while ((g = __atomic_load_n(&gen_, __ATOMIC_ACQUIRE)) == seen) combine_futex_wait(&gen_, seen, 0, false);
            seen = g;
            if (__atomic_load_n(&quit_, __ATOMIC_ACQUIRE)) return;
            work();
            if (__atomic_sub_fetch(&busy_, 1u, __ATOMIC_ACQ_REL) == 0) combine_futex_wake(&busy_, false);
        }
    }
    std::vector<std::thread> workers_;
    uint32_t gen_ = 0, next_ = 0, busy_ = 0, quit_ = 0;
    const uint16_t* src_[r3::moe_shm::kCombineHelpers] = {};
    uint32_t nsrc_ = 0;
    uint16_t* out_ = nullptr;
    const uint8_t* mask_ = nullptr;
    uint32_t rows_ = 0, hidden_ = 0;
};

struct CombineStats {
    uint64_t steps = 0;
    double combine_us = 0.0;
};

/* Helper 0, right after queueing its own large-lane request q0 (before its stream sync): if the engine asked for a combine
 * on this step, combine chunk by chunk (RAD42). For each chunk of kCombineChunkRows rows it waits until every helper with a
 * request this step, helper 0 included, has published the chunk (Header::chunks.seq, stored by their reduce kernels), sums
 * it into plane [0][1] and publishes chunks.done[c]; the engine's add kernel takes each chunk as it lands. Once every chunk
 * is out it checks the other helpers' requests completed and publishes combine.done. The waits spin with a yield: they last
 * only while this step's helpers run (helper 0's own GPU is busy then too), never between bursts. False on a protocol
 * mismatch, a helper fault or a stop. */
inline bool combine_step(void* map, const r3::moe_shm::Request* q0, uint32_t helpers, CombinePool* pool,
                         std::vector<uint8_t>* mask, CombineStats* st) {
    using namespace r3::moe_shm;
    auto* h = static_cast<Header*>(map);
    const uint32_t g = q0->combine_generation;
    if (!g) return true;
    const uint32_t rows = h->combine.rows;
    if (helpers < 2 || helpers > kCombineHelpers || rows == 0 || rows > h->max_rows || q0->rows != rows ||
        h->combine.seq[0] != q0->sequence) {
        std::fprintf(stderr, "r3_moe_worker: combine stopped: the combine request does not match helper 0's request\n");
        return false;
    }
    const Request* q[kCombineHelpers] = {q0};
    uint32_t seq[kCombineHelpers] = {q0->sequence};
    for (uint32_t k = 1; k < helpers; ++k) {
        const uint32_t lanek = h->combine.lane[k], seqk = h->combine.seq[k];
        if (!seqk) continue;   /* no request from helper k this step: it contributes no rows */
        const Request* qk = lane_request(map, lanek, seqk);
        if (lanek >= kLanes || lanek % kMaxSlots != k || !qk) {
            std::fprintf(stderr, "r3_moe_worker: combine stopped: helper %u's lane does not match\n", k);
            return false;
        }
        /* This step's requests were all written before any was released, but helper 0 reads helper k's from another
         * card's writes: wait until they are visible here, then check them. */
        while (load(&qk->sequence) != seqk) {
            if (stopping() || load(&h->helper_fault)) return false;
            std::this_thread::yield();
        }
        if (qk->rows != rows) {
            std::fprintf(stderr, "r3_moe_worker: combine stopped: helper %u's request does not match\n", k);
            return false;
        }
        q[k] = qk;
        seq[k] = seqk;
    }
    /* Which helpers wrote which rows: from the route lists, which the engine wrote before releasing the requests. */
    std::memset(mask->data(), 0, rows);
    const uint16_t* planes[kCombineHelpers] = {};
    for (uint32_t k = 0; k < helpers; ++k) {
        planes[k] = ptr<uint16_t>(map, h->result_offset[result_index(k, 0)]);
        if (!q[k]) continue;
        const Route* r = ptr<Route>(map, q[k]->route_offset);
        for (uint32_t j = 0; j < q[k]->route_count; ++j) {
            const int32_t t = r[j].token;
            if (t < 0 || (uint32_t)t >= rows) {
                std::fprintf(stderr, "r3_moe_worker: combine stopped: a route token outside the step\n");
                return false;
            }
            (*mask)[(uint32_t)t] |= (uint8_t)(1u << k);
        }
    }
    uint16_t* out = ptr<uint16_t>(map, h->result_offset[result_index(0, 1)]);
    const uint32_t hid = h->hidden;
    std::chrono::steady_clock::duration waited{}, combined{};
    for (uint32_t c = 0, r0 = 0; r0 < rows; ++c, r0 += kCombineChunkRows) {
        const uint32_t n = std::min(kCombineChunkRows, rows - r0);
        const auto tw = std::chrono::steady_clock::now();
        for (uint32_t k = 0; k < helpers; ++k) {
            if (!q[k]) continue;
            while (load(&h->chunks.seq[k][c]) != seq[k]) {
                if (stopping() || load(&h->helper_fault)) return false;
                std::this_thread::yield();
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        const uint16_t* src[kCombineHelpers] = {};
        for (uint32_t k = 0; k < helpers; ++k) src[k] = planes[k] + (uint64_t)r0 * hid;
        pool->combine(src, helpers, out + (uint64_t)r0 * hid, mask->data() + r0, n, hid);
        std::atomic_thread_fence(std::memory_order_release);
        store(&h->chunks.done[c], g);
        const auto t1 = std::chrono::steady_clock::now();
        waited += t0 - tw;
        combined += t1 - t0;
    }
    /* every other helper's request finished cleanly (its rows are all in; its done word follows its last chunk) */
    for (uint32_t k = 1; k < helpers; ++k) {
        if (!q[k]) continue;
        const uint32_t lanek = h->combine.lane[k];
        while (load(&h->lane[lanek].done_sequence) != seq[k]) {
            if (stopping() || load(&h->helper_fault)) return false;
            std::this_thread::yield();
        }
        if (q[k]->status != 0) {
            std::fprintf(stderr, "r3_moe_worker: combine stopped: helper %u's request failed (%d)\n", k, q[k]->status);
            return false;
        }
    }
    store(&h->combine.done, g);
    st->combine_us += std::chrono::duration<double, std::micro>(combined).count();
    {   /* R3_MOE_WORKER_TRACE (RAD40): this step's waits for chunks and its combine time, on helper 0's request */
        const auto us = [](auto d) { return (uint64_t)std::min<int64_t>(UINT32_MAX, std::chrono::duration_cast<std::chrono::microseconds>(d).count()); };
        const_cast<Request*>(q0)->stage_trace[2] = us(waited) | (us(combined) << 32);
    }
    ++st->steps;
    return true;
}

} // namespace
