#pragma once

// R3_MOE_WORKER_COMBINE (RAD18): helper 0 adds the two helpers' bf16 rows on the CPU so the one-rank engine reads one row
// a token over its link instead of two. The sum is written as fp16: the same two bytes as one bf16 row, but rounded 8x
// finer, since a bf16 sum measurably worsened KLD (RAD18 2026-10-23: p99 +0.275). A row with a value outside fp16 range
// is marked kCombineFallback in its first element and the engine reads that row from the two helper planes instead.
// Nothing here spins: helper 0 combines on its large-lane thread right after its own stream sync, futex-waiting on helper
// 1's host done word; the pool's workers sleep on futex words between jobs.
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
    /* mask[t]: bit 0 helper 0 wrote row t, bit 1 helper 1 did. Rows with neither are not read by the engine. */
    void combine(const uint16_t* a, const uint16_t* b, uint16_t* out, const uint8_t* mask, uint32_t rows, uint32_t hidden) {
        a_ = a; b_ = b; out_ = out; mask_ = mask; rows_ = rows; hidden_ = hidden;
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
                if (combine_row_f16(a_ + at, b_ + at, m, out_ + at, hidden_)) out_[at] = r3::moe_shm::kCombineFallback;
            }
        }
    }
    void run() {
        uint32_t seen = __atomic_load_n(&gen_, __ATOMIC_ACQUIRE);
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
    const uint16_t* a_ = nullptr;
    const uint16_t* b_ = nullptr;
    uint16_t* out_ = nullptr;
    const uint8_t* mask_ = nullptr;
    uint32_t rows_ = 0, hidden_ = 0;
};

struct CombineStats {
    uint64_t steps = 0;
    double combine_us = 0.0;
};

/* Helper 1, after its large-lane request seq has synced: tell helper 0. */
inline void combine_helper1_done(void* map, uint32_t seq) {
    auto* h = static_cast<r3::moe_shm::Header*>(map);
    r3::moe_shm::store(&h->combine.helper1_done, seq);
    combine_futex_wake(&h->combine.helper1_done, true);
}

/* Helper 0, after its large-lane request q has synced: if the engine asked for a combine on this step, wait for helper 1
 * (futex), combine into plane [0][1] and publish done. False on a protocol mismatch or a stop. */
inline bool combine_step(void* map, const r3::moe_shm::Request* q0, CombinePool* pool, std::vector<uint8_t>* mask,
                         CombineStats* st) {
    using namespace r3::moe_shm;
    auto* h = static_cast<Header*>(map);
    const uint32_t g = q0->combine_generation;
    if (!g) return true;
    const uint32_t rows = h->combine.rows, lane1 = h->combine.lane[1], seq1 = h->combine.seq[1];
    if (rows == 0 || rows > h->max_rows || q0->rows != rows || h->combine.seq[0] != q0->sequence ||
        lane1 >= kLanes || lane1 % kMaxRanks != 1u || !seq1) {
        std::fprintf(stderr, "r3_moe_worker: combine stopped: the combine request does not match helper 0's request\n");
        return false;
    }
    for (uint32_t d; (int32_t)((d = load(&h->combine.helper1_done)) - seq1) < 0;) {
        if (stopping() || load(&h->helper_fault)) return false;
        combine_futex_wait(&h->combine.helper1_done, d, 200, true);
    }
    const Request* q1 = lane_request(map, lane1, seq1);
    if (load(&h->lane[lane1].done_sequence) != seq1 || !q1 || load(&q1->sequence) != seq1 || q1->rows != rows ||
        q1->status != 0) {
        std::fprintf(stderr, "r3_moe_worker: combine stopped: helper 1's request does not match\n");
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::memset(mask->data(), 0, rows);
    const Request* q[kMaxRanks] = {q0, q1};
    for (uint32_t k = 0; k < kMaxRanks; ++k) {
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
    pool->combine(ptr<uint16_t>(map, h->result_offset[0 * kMaxRanks + 0]),
                  ptr<uint16_t>(map, h->result_offset[1 * kMaxRanks + 1]),
                  ptr<uint16_t>(map, h->result_offset[0 * kMaxRanks + 1]), mask->data(), rows, h->hidden);
    std::atomic_thread_fence(std::memory_order_release);
    store(&h->combine.done, g);
    st->combine_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    ++st->steps;
    return true;
}

} // namespace
