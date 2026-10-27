#pragma once
// Included only by r3_moe_worker.cpp, after worker_container.h, worker_combine.h and worker_harness.h.
//
// The GPU serving loop: one thread per lane this helper serves. A small lane pre-queues kQueueDepth requests at its
// fixed row count; a large lane is sized to each request on arrival. RAD20: with power save, small lanes arm their
// pre-queued waits only inside an engine burst (Header::burst odd) and release and drain them when it ends (park),
// so no wait is pending between bursts and the GPU idles; a pending wait keeps a gfx1100 MEC busy at ~95 W (FINDINGS).
//
// Lifetime: the Lanes object references the segment, the GPU runner and the combine state, all of which outlive it.
// stop_and_join() publishes the stop (fault word, burst stop bit, park flags, wakes) before joining, so no thread can
// stay blocked on a futex or a drain.
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

/* Block until the engine is inside an inference burst (Header::burst odd). False once this helper is stopping. */
inline bool wait_burst(r3::moe_shm::Header* h) {
    using namespace r3::moe_shm;
    for (;;) {
        const uint32_t b = load(&h->burst);
        if ((b & kBurstStop) || stopping()) return false;
        if (b & 1u) return true;
        if (futex_wait(&h->burst, b) < 0 && errno != EINTR && errno != EAGAIN) {
            std::fprintf(stderr, "r3_moe_worker: futex_wait(burst): %s\n", std::strerror(errno));
            request_stop();
            return false;
        }
    }
}

/* The engine's large-lane rows added on the CPU by helper 0 (RAD18, worker_combine.h); helper 1 signals its sync. */
struct Combine {
    bool on = false;
    std::unique_ptr<CombinePool> pool;   /* helper 0 only */
    std::vector<uint8_t> mask;
    CombineStats stats;
};

class Lanes {
public:
    static constexpr uint32_t kQueueDepth = 4;

    Lanes(r3::moe_shm::Header* h, void* map, void* dev_map, r3_moe_gpu::Runner& gpu, const SmallClasses& classes,
          const WorkerConfig& cfg, const std::array<bool, r3::moe_shm::kLanes>& served, Combine& combine)
        : h_(h), map_(map), dev_map_(dev_map), gpu_(gpu), classes_(classes), cfg_(cfg), served_(served),
          combine_(combine) {}
    Lanes(const Lanes&) = delete;
    Lanes& operator=(const Lanes&) = delete;
    /* Any path out of the serving scope stops the lanes first: a plain join could wait on a lane still serving. */
    ~Lanes() {
        for (const auto& t : threads_)
            if (t.joinable()) { stop_and_join(); break; }
    }

    /* One thread per served lane. A thread that cannot start stops and joins the ones that did. */
    bool start() {
        try {
            for (uint32_t lane = 0; lane < r3::moe_shm::kLanes; ++lane)
                if (served_[lane]) threads_[lane] = std::thread([this, lane] { run(lane); });
        } catch (const std::exception& e) {
            std::fprintf(stderr, "r3_moe_worker: lane thread: %s\n", e.what());
            stop_and_join();
            return false;
        }
        return true;
    }

    /* Stopping (a signal or a failed lane): fail the engine's waits and wake every waiter of this helper before
     * joining anything, so no fault path depends on a join completing. */
    void stop_and_join() {
        using namespace r3::moe_shm;
        request_stop();
        combine_futex_wake(&h_->combine.helper1_done, true);
        (void)__atomic_fetch_or(&h_->burst, kBurstStop, __ATOMIC_ACQ_REL);
        (void)futex_wake(&h_->burst);
        for (uint32_t lane = 0; lane < kLanes; ++lane)
            if (served_[lane]) store(&h_->lane[lane].park, 0u);
        raise_fault(h_);   /* after the park flags; wakes every engine CPU wait (Header::notify) */
        join();
    }

    /* After stop_and_join(): the per-lane, per-class and timing report; returns requests/rows/routes/us totals. */
    void report(uint64_t* req, uint64_t* rows, uint64_t* routes, double* us) const {
        using namespace r3::moe_shm;
        uint64_t samples = 0;
        double sampled_us = 0.0;
        for (uint32_t lane = 0; lane < kLanes; ++lane) {
            *req += stats_[lane].req;
            *rows += stats_[lane].rows;
            *routes += stats_[lane].routes;
            samples += stats_[lane].samples;
            sampled_us += stats_[lane].sample_us;
            gpu_.stage_report(lane, stats_[lane].samples);
        }
        for (uint32_t c = 0; c < classes_.count; ++c) {
            uint64_t r = 0, n = 0;
            for (uint32_t k = 0; k < kMaxRanks; ++k) { r += stats_[lane_of(k, c)].req; n += stats_[lane_of(k, c)].rows; }
            std::fprintf(stderr, "r3_moe_worker: small lane class %u (rows <= %u): requests %llu (%llu rows)\n", c,
                         classes_.rows[c], (unsigned long long)r, (unsigned long long)n);
        }
        uint64_t lr = 0, ln = 0;
        for (uint32_t k = 0; k < kMaxRanks; ++k) { lr += stats_[lane_of(k, kSmallClasses)].req; ln += stats_[lane_of(k, kSmallClasses)].rows; }
        std::fprintf(stderr, "r3_moe_worker: large-lane requests %llu (%llu rows), notify wakes %llu\n",
                     (unsigned long long)lr, (unsigned long long)ln,
                     (unsigned long long)large_wakes_.load(std::memory_order_relaxed));
        if (samples && *req) *us = (sampled_us / (double)samples) * (double)*req;
        std::fprintf(stderr, "r3_moe_worker: GPU timing samples=%llu sampled=%.1f ms avg=%.1f us\n",
                     (unsigned long long)samples, sampled_us / 1000.0, samples ? sampled_us / (double)samples : 0.0);
    }

private:
    struct Stats { uint64_t req = 0, rows = 0, routes = 0, samples = 0; double sample_us = 0.0; };

    void join() {
        for (auto& t : threads_) if (t.joinable()) t.join();
    }
    /* A lane thread's entry: nothing escapes it; any failure stops the helper. */
    void run(uint32_t lane) noexcept {
        try {
            r3::moe_shm::lane_is_small(lane) ? run_small(lane) : run_large(lane);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "r3_moe_worker: lane %u: %s\n", lane, e.what());
            request_stop();
        } catch (...) {
            std::fprintf(stderr, "r3_moe_worker: lane %u: unknown exception\n", lane);
            request_stop();
        }
    }
    bool collect(uint32_t lane, uint32_t done) {
        return gpu_.collect_lane_timing(lane, done, &stats_[lane].sample_us, &stats_[lane].samples);
    }
    /* Every lane ends here: drain the stream and read the device-side counters. */
    void finish(uint32_t lane) {
        uint64_t s[3]{};
        if (!gpu_.sync_lane(lane) || !gpu_.lane_stats(lane, s)) { request_stop(); return; }
        stats_[lane].req = s[0];
        stats_[lane].rows = s[1];
        stats_[lane].routes = s[2];
    }

    /* Large lane: wait for the engine's request word, read its rows, launch that size. The engine publishes from GPU
     * code, and a host callback queued behind that publish bumps Header::notify (RAD26), so inside a burst this
     * thread sleeps on the notify generation; between bursts it sleeps on the burst word. */
    void run_large(uint32_t lane) {
        using namespace r3::moe_shm;
        uint32_t* const req = &h_->lane[lane].request_sequence;
        for (uint32_t next = 1; !stopping();) {
            if (seq_after(next, load(req))) {
                if (!(load(&h_->burst) & 1u)) {
                    if (!wait_burst(h_)) break;
                    continue;
                }
                /* Woken by the engine's publish callback, the burst ending, or a stop/fault (raise_fault). */
                const WaitResult r = wait_for(h_, nullptr, [&] {
                    return !seq_after(next, load(req)) || !(load(&h_->burst) & 1u) || stopping();
                });
                if (r != WaitResult::Ready) { request_stop(); break; }   /* a helper failed, or the wait itself did */
                large_wakes_.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            const Request* q = lane_request(map_, lane, next);
            const uint32_t rows = load(&q->rows);
            capture_request(map_, q, lane, next);
            trace_dump(map_, lane, next - 1u);
            /* Every large request is timed by stage: they are few, and the engine cannot send the next one until it
             * has merged this one, so waiting here costs nothing. */
            if (const int rc = gpu_.enqueue_lane(map_, dev_map_, lane, next, rows, true); rc != 0) {
                std::fprintf(stderr, "r3_moe_worker: large lane %u seq %u rows %u failed rc=%d\n", lane, next, rows, rc);
                request_stop();
                break;
            }
            if (!gpu_.sync_lane(lane) || !collect(lane, next)) { request_stop(); break; }
            if (combine_.on && cfg_.helper_index == 1) combine_helper1_done(map_, next);
            if (combine_.on && cfg_.helper_index == 0 &&
                !combine_step(map_, q, combine_.pool.get(), &combine_.mask, &combine_.stats)) {
                raise_fault(h_);   /* the engine's wait for combine.done fails now */
                request_stop();
                break;
            }
            if (++next == 0) {
                std::fprintf(stderr, "r3_moe_worker: large lane %u sequence space exhausted\n", lane);
                request_stop();
                break;
            }
        }
        finish(lane);
    }

    void run_small(uint32_t lane) {
        using namespace r3::moe_shm;
        const uint32_t rows = classes_.rows[lane_class(lane)];
        uint32_t next = 1, done_seen = 0;
        auto queue_one = [&]() -> bool {
            if (next == 0) {
                std::fprintf(stderr, "r3_moe_worker: lane %u sequence space exhausted\n", lane);
                return false;
            }
            if (const int rc = gpu_.enqueue_lane(map_, dev_map_, lane, next, rows, (next & 63u) == 0); rc != 0) {
                std::fprintf(stderr, "r3_moe_worker: lane %u enqueue seq %u failed rc=%d\n", lane, next, rc);
                return false;
            }
            ++next;
            if (next == 0) {   /* the last sequence is queued; fail now rather than on the next call */
                std::fprintf(stderr, "r3_moe_worker: lane %u sequence space exhausted\n", lane);
                return false;
            }
            return true;
        };
        auto fill = [&]() -> bool {
            while (!stopping() && (uint32_t)(next - done_seen - 1u) < kQueueDepth)
                if (!queue_one()) return false;
            return true;
        };
        uint32_t* park = &h_->lane[lane].park;
        bool armed = false;
        while (wait_burst(h_)) {
            if (!armed) {
                next = done_seen + 1u;
                if (!fill()) { request_stop(); break; }
                armed = true;
            }
            /* Serve the burst: collect each completion, keep kQueueDepth requests queued. */
            while (!stopping()) {
                const uint32_t done = load(&h_->lane[lane].done_sequence);
                if (!seq_after(done, done_seen)) {
                    if (!(load(&h_->burst) & 1u)) break;   /* the burst ended */
                    std::this_thread::yield();
                    continue;
                }
                if (!collect(lane, done)) { request_stop(); break; }
                done_seen = done;
                trace_dump(map_, lane, done - 1u);
                if (!fill()) { request_stop(); break; }
            }
            if (stopping()) break;
            if (!cfg_.power_save) continue;   /* stay armed; wait_burst sleeps until the next burst */
            const Park p = park_lane(lane, park, next, &done_seen);
            if (p == Park::Failed) break;
            armed = p == Park::Reopened;
        }
        collect(lane, load(&h_->lane[lane].done_sequence));
        /* Stopping: release the pre-queued waits so the stream drains. */
        store(&h_->lane[lane].request_sequence, next - 1u);
        finish(lane);
    }

    enum class Park { Parked, Reopened, Failed };

    /* Park a small lane after its burst ended (Header::burst describes the handshake). Flag the drain, then re-check:
     * the engine opens a burst before it reads the flag, so either it waits for the drain or we see its burst and
     * stay armed (Reopened). The engine is idle and has every result, so each request it published is done. Release
     * the unconsumed waits (their ring slots hold older sequences, so the kernels see an invalid request, and the
     * publish limit stops them publishing), drain, restore the engine's last sequence and the limit, clear the flag.
     * Failed when stopping or on a GPU error. */
    Park park_lane(uint32_t lane, uint32_t* park, uint32_t next, uint32_t* done_seen) {
        using namespace r3::moe_shm;
        __atomic_store_n(park, kParkDraining, __ATOMIC_SEQ_CST);
        test_stop_at("park_drain", cfg_.helper_index);
        if (__atomic_load_n(&h_->burst, __ATOMIC_SEQ_CST) & 1u) {
            store(park, 0u);
            notify(h_);   /* Header::notify: the engine sleeps on the generation word */
            return Park::Reopened;
        }
        const uint32_t last = load(&h_->lane[lane].request_sequence);
        while (!stopping() && load(&h_->lane[lane].done_sequence) != last) std::this_thread::yield();
        if (stopping()) return Park::Failed;
        if (!collect(lane, last) || !gpu_.set_publish_limit(lane, last)) { request_stop(); return Park::Failed; }
        store(&h_->lane[lane].request_sequence, next - 1u);
        if (!gpu_.sync_lane(lane) || !gpu_.set_publish_limit(lane, UINT32_MAX)) { request_stop(); return Park::Failed; }
        store(&h_->lane[lane].request_sequence, last);
        gpu_.drop_lane_sample(lane);
        *done_seen = last;
        store(park, 0u);
        notify(h_);   /* Header::notify: the engine sleeps on the generation word */
        return Park::Parked;
    }

    r3::moe_shm::Header* h_;
    void* map_;
    void* dev_map_;
    r3_moe_gpu::Runner& gpu_;
    const SmallClasses& classes_;
    const WorkerConfig& cfg_;
    const std::array<bool, r3::moe_shm::kLanes>& served_;
    Combine& combine_;
    std::array<std::thread, r3::moe_shm::kLanes> threads_;
    std::array<Stats, r3::moe_shm::kLanes> stats_{};
    std::atomic<uint64_t> large_wakes_{0};   /* RAD26 diagnostics: large-lane notify wakes inside bursts */
};
