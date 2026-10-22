#pragma once

// R3_MOE_WORKER_COMBINE (RAD18): helper 0 adds the two helpers' bf16 rows on the CPU so the one-rank engine reads one row
// a token over its link instead of two. Included only by r3_moe_worker.cpp, after worker_container.h (g_stop).

namespace {

inline float combine_bf16_f32(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; std::memcpy(&f, &u, 4); return f; }
inline uint16_t combine_f32_bf16_rn(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

/* One coordinator (the calling thread) plus persistent workers; a job is split into token chunks taken from a counter. */
class CombinePool {
public:
    explicit CombinePool(uint32_t threads) {
        for (uint32_t i = 1; i < threads; ++i) workers_.emplace_back([this] { run(); });
    }
    ~CombinePool() {
        quit_.store(true, std::memory_order_release);
        for (auto& t : workers_) t.join();
    }
    /* mask[t]: bit 0 helper 0 wrote row t, bit 1 helper 1 did. Rows with neither are not read by the engine. */
    void combine(const uint16_t* a, const uint16_t* b, uint16_t* out, const uint8_t* mask, uint32_t rows, uint32_t hidden) {
        a_ = a; b_ = b; out_ = out; mask_ = mask; rows_ = rows; hidden_ = hidden;
        next_.store(0, std::memory_order_relaxed);
        busy_.store((uint32_t)workers_.size(), std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        work();
        while (busy_.load(std::memory_order_acquire) != 0) {}
    }

private:
    static constexpr uint32_t kChunk = 32;
    void work() {
        for (;;) {
            const uint32_t t0 = next_.fetch_add(kChunk, std::memory_order_relaxed);
            if (t0 >= rows_) return;
            const uint32_t t1 = std::min(rows_, t0 + kChunk);
            for (uint32_t t = t0; t < t1; ++t) {
                const uint8_t m = mask_[t];
                const uint64_t at = (uint64_t)t * hidden_;
                if (m == 3u) {
                    for (uint32_t c = 0; c < hidden_; ++c)
                        out_[at + c] = combine_f32_bf16_rn(combine_bf16_f32(a_[at + c]) + combine_bf16_f32(b_[at + c]));
                } else if (m == 1u) {
                    std::memcpy(out_ + at, a_ + at, (size_t)hidden_ * 2u);
                } else if (m == 2u) {
                    std::memcpy(out_ + at, b_ + at, (size_t)hidden_ * 2u);
                }
            }
        }
    }
    void run() {
        uint32_t seen = 0;
        while (!quit_.load(std::memory_order_acquire)) {
            const uint32_t g = gen_.load(std::memory_order_acquire);
            if (g == seen) { std::this_thread::yield(); continue; }
            seen = g;
            work();
            busy_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
    std::vector<std::thread> workers_;
    std::atomic<uint32_t> gen_{0}, next_{0}, busy_{0};
    std::atomic<bool> quit_{false};
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

/* Serve combine requests until g_stop. Any protocol mismatch stops the helper (helper_fault), never a stale merge. */
inline void run_combiner(void* map, uint32_t threads, CombineStats* st) {
    using namespace r3::moe_shm;
    auto* h = static_cast<Header*>(map);
    CombinePool pool(threads);
    std::vector<uint8_t> mask(h->max_rows);
    uint32_t last = load(&h->combine.done);
    auto fail = [&](const char* why) {
        std::fprintf(stderr, "r3_moe_worker: combine stopped: %s\n", why);
        g_stop = 1;
    };
    while (!g_stop) {
        const uint32_t g = load(&h->combine.request);
        if (g == last) { std::this_thread::yield(); continue; }
        const uint32_t rows = h->combine.rows;
        if (rows == 0 || rows > h->max_rows) { fail("rows outside the segment's capacity"); return; }
        const Request* q[kMaxRanks] = {};
        for (uint32_t k = 0; k < kMaxRanks; ++k) {
            const uint32_t lane = h->combine.lane[k], seq = h->combine.seq[k];
            if (lane >= kLanes || lane % kMaxRanks != k || !seq) { fail("lane/sequence outside the protocol"); return; }
            while (load(&h->lane[lane].done_sequence) != seq) {
                if (g_stop) return;
                if (load(&h->helper_fault)) { fail("a helper faulted"); return; }
                std::this_thread::yield();
            }
            q[k] = lane_request(map, lane, seq);
            if (!q[k] || load(&q[k]->sequence) != seq || q[k]->rows != rows || q[k]->hidden != h->hidden ||
                q[k]->status != 0) { fail("request does not match the combine request"); return; }
        }
        const auto t0 = std::chrono::steady_clock::now();
        std::memset(mask.data(), 0, rows);
        for (uint32_t k = 0; k < kMaxRanks; ++k) {
            const Route* r = ptr<Route>(map, q[k]->route_offset);
            for (uint32_t j = 0; j < q[k]->route_count; ++j) {
                const int32_t t = r[j].token;
                if (t < 0 || (uint32_t)t >= rows) { fail("route token outside the step"); return; }
                mask[(uint32_t)t] |= (uint8_t)(1u << k);
            }
        }
        pool.combine(ptr<uint16_t>(map, h->result_offset[0 * kMaxRanks + 0]),
                     ptr<uint16_t>(map, h->result_offset[1 * kMaxRanks + 1]),
                     ptr<uint16_t>(map, h->result_offset[0 * kMaxRanks + 1]), mask.data(), rows, h->hidden);
        std::atomic_thread_fence(std::memory_order_release);
        store(&h->combine.done, g);
        st->combine_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        ++st->steps;
        last = g;
    }
}

} // namespace
