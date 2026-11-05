/* RHP01: read-only evidence for the existing Ctx histogram ring.
 * Diagnostic only (RAD_ROUTING_EVIDENCE=1). No GPU operations, allocations or host
 * copies when disabled. The caller owns Ctx and serialises issue/drain per rank.
 * The registry lock protects only lookup/lifetime; a rank's tracker is single-writer.
 */
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

struct BcMoeRemoteEvidence {
    enum Phase { Prefill = 0, Decode = 1, Mixed = 2, Draft = 3, Phases = 4 };
    struct Counts {
        uint64_t issued = 0, produced = 0, completed = 0, credited = 0, overwritten = 0;
        uint64_t produced_passes = 0, credited_passes = 0, lost_passes = 0;
    };
    struct Slot {
        int step = -1, phase = Prefill, passes = 0;
        uint64_t generation = 0;
        bool flushed = false, completed = false, credited = false, dropped = false;
    };
    int layers = 0, banks = 0, latest_step = -1;
    uint64_t generation = 0, outstanding = 0, high_water = 0;
    uint64_t stale_event_rejected = 0, duplicate_rejected = 0, order_rejected = 0, overflow = 0;
    Counts count[Phases]{};
    std::vector<Slot> slots;
    std::vector<int> last_credited;
    uint64_t delay_bins[65]{};
    int last_loss_layer = -1, last_loss_step = -1, last_loss_phase = -1;

    static void add(uint64_t& v, uint64_t n, uint64_t& overflow_count) {
        if (std::numeric_limits<uint64_t>::max() - v < n) {
            v = std::numeric_limits<uint64_t>::max();
            if (overflow_count != std::numeric_limits<uint64_t>::max()) ++overflow_count;
        } else v += n;
    }
    void plus(uint64_t& v, uint64_t n = 1) { add(v, n, overflow); }
    void reset(int n_layers, int n_banks) {
        layers = n_layers;
        banks = n_banks;
        latest_step = -1;
        generation = outstanding = high_water = 0;
        stale_event_rejected = duplicate_rejected = order_rejected = overflow = 0;
        for (auto& c : count) c = Counts{};
        slots.assign((size_t)layers * (size_t)banks, Slot{});
        last_credited.assign((size_t)layers, -1);
        std::fill(std::begin(delay_bins), std::end(delay_bins), 0);
        last_loss_layer = last_loss_step = last_loss_phase = -1;
    }
    Slot* slot(int layer, int bank) {
        if (layer < 0 || layer >= layers || bank < 0 || bank >= banks) return nullptr;
        return &slots[(size_t)layer * (size_t)banks + (size_t)bank];
    }
    static int phase(int64_t n_seq, int64_t n_decode, int draft_pass) {
        if (draft_pass) return Draft;
        return n_decode <= 0 ? Prefill : n_decode >= n_seq ? Decode : Mixed;
    }
    void lose(Slot& s, int layer) {
        if (s.step < 0 || s.dropped || s.credited) return;
        plus(count[s.phase].overwritten);
        plus(count[s.phase].lost_passes, (uint64_t)s.passes);
        if (outstanding) --outstanding;
        last_loss_step = s.step;
        last_loss_layer = layer;
        last_loss_phase = s.phase;
        s.dropped = true;
    }
    void issue(int layer, int bank, int step, int kind) {
        Slot* s = slot(layer, bank);
        if (!s || step < 0 || kind < 0 || kind >= Phases) return;
        latest_step = std::max(latest_step, step);
        if (s->step == step && !s->dropped && !s->credited) {
            if (s->phase != kind) { plus(duplicate_rejected); return; }
            if (s->passes == std::numeric_limits<int>::max()) { plus(overflow); return; }
            ++s->passes;
        } else {
            if (s->step == step) { plus(duplicate_rejected); return; }
            lose(*s, layer);
            *s = Slot{};
            s->step = step;
            s->phase = kind;
            s->passes = 1;
            s->generation = ++generation;
            plus(count[kind].issued);
            plus(outstanding);
            high_water = std::max(high_water, outstanding);
        }
        plus(count[kind].produced);
        plus(count[kind].produced_passes);
        /* Same-step later pass invalidates previous event completion until its flush is enqueued. */
        s->flushed = false;
        s->completed = false;
    }
    void flush(int bank, int lo, int hi, int step) {
        if (bank < 0 || bank >= banks || lo > hi) return;
        for (int layer = std::max(0, lo); layer <= hi && layer < layers; ++layer) {
            Slot* s = slot(layer, bank);
            if (!s) continue;
            if (s->step != step) {
                /* The original contiguous DMA also copies untouched rows in [lo,hi]. */
                lose(*s, layer);
            } else if (!s->dropped && !s->credited) {
                s->flushed = true;
            }
        }
    }
    bool landed(int layer, int bank, int step, bool event_ready) {
        if (!event_ready) return false;
        Slot* s = slot(layer, bank);
        if (!s || s->step != step || !s->flushed || s->dropped || s->credited) {
            plus(stale_event_rejected);
            return false;
        }
        if (!s->completed) {
            s->completed = true;
            plus(count[s->phase].completed);
        }
        for (int b = 0; b < banks; ++b) {
            const Slot* earlier = slot(layer, b);
            if (earlier && earlier->step >= 0 && earlier->step < step &&
                !earlier->credited && !earlier->dropped) return false;
        }
        return true;
    }
    bool credit(int layer, int bank, int step, int passes) {
        Slot* s = slot(layer, bank);
        if (!s || s->step != step || s->credited || s->dropped ||
            !s->completed || s->passes != passes) {
            plus(duplicate_rejected);
            return false;
        }
        if (last_credited[(size_t)layer] >= step) {
            plus(order_rejected);
            return false;
        }
        last_credited[(size_t)layer] = step;
        s->credited = true;
        if (outstanding) --outstanding;
        plus(count[s->phase].credited);
        plus(count[s->phase].credited_passes, (uint64_t)passes);
        const int age = std::max(0, latest_step - step);
        plus(delay_bins[(size_t)std::min(age, 64)]);
        return true;
    }
    int percentile(unsigned pct) const {
        uint64_t total = 0, accum = 0;
        for (uint64_t v : delay_bins) {
            if (std::numeric_limits<uint64_t>::max() - total < v) return 64;
            total += v;
        }
        if (!total) return -1;
        const uint64_t threshold = total / 100 * pct + (total % 100 * pct + 99) / 100;
        for (int i = 0; i <= 64; ++i) {
            accum += delay_bins[i];
            if (accum >= threshold) return i;
        }
        return 64;
    }
    void print(int rank, uint64_t upstream_dropped) const {
        static const char* names[] = {"prefill", "decode", "mixed", "draft"};
        for (int k = 0; k < Phases; ++k) {
            uint64_t pending = 0, in_flight = 0;
            int oldest = -1, oldest_step = -1;
            for (const auto& s : slots) {
                if (s.phase != k || s.step < 0 || s.credited || s.dropped) continue;
                ++pending;
                if (!s.completed) ++in_flight;
                oldest = std::max(oldest, latest_step - s.step);
                if (oldest_step < 0 || s.step < oldest_step) oldest_step = s.step;
            }
            const Counts& c = count[k];
            std::fprintf(stderr,
                "RHP01 rank=%d phase=%s issued=%llu produced=%llu completed=%llu credited=%llu "
                "overwritten=%llu pending=%llu in_flight=%llu produced_passes=%llu "
                "credited_passes=%llu lost_passes=%llu oldest_uncredited_steps=%d oldest_uncredited_source_step=%d "
                "latest_step=%d high_water=%llu delay_steps_p50=%d delay_steps_p95=%d\n",
                rank, names[k], (unsigned long long)c.issued, (unsigned long long)c.produced,
                (unsigned long long)c.completed, (unsigned long long)c.credited,
                (unsigned long long)c.overwritten, (unsigned long long)pending,
                (unsigned long long)in_flight, (unsigned long long)c.produced_passes,
                (unsigned long long)c.credited_passes, (unsigned long long)c.lost_passes,
                oldest, oldest_step, latest_step, (unsigned long long)high_water,
                percentile(50), percentile(95));
        }
        std::fprintf(stderr,
            "RHP01 rank=%d ring_banks=%d stale_event_rejected=%llu duplicate_rejected=%llu "
            "order_rejected=%llu counter_overflow=%llu upstream_dropped=%llu "
            "last_loss_step=%d last_loss_layer=%d last_loss_phase=%d "
            "coverage_basis=layer_histograms source=existing-route-ring\n",
            rank, banks, (unsigned long long)stale_event_rejected,
            (unsigned long long)duplicate_rejected, (unsigned long long)order_rejected,
            (unsigned long long)overflow, (unsigned long long)upstream_dropped,
            last_loss_step, last_loss_layer, last_loss_phase);
    }
};

inline bool bc_moe_remote_evidence_enabled() {
    static const bool enabled = [] {
        const char* s = std::getenv("RAD_ROUTING_EVIDENCE");
        return s && s[0] == '1' && s[1] == '\0';
    }();
    return enabled;
}
inline std::mutex& bc_moe_remote_evidence_mutex() {
    static std::mutex m;
    return m;
}
inline std::map<const void*, std::unique_ptr<BcMoeRemoteEvidence>>& bc_moe_remote_evidence_registry() {
    static std::map<const void*, std::unique_ptr<BcMoeRemoteEvidence>> r;
    return r;
}
inline BcMoeRemoteEvidence* bc_moe_remote_evidence_get(const void* ctx, bool create = false) {
    if (!bc_moe_remote_evidence_enabled()) return nullptr;
    std::lock_guard<std::mutex> lock(bc_moe_remote_evidence_mutex());
    auto& registry = bc_moe_remote_evidence_registry();
    auto it = registry.find(ctx);
    if (it != registry.end()) return it->second.get();
    if (!create) return nullptr;
    return (registry[ctx] = std::make_unique<BcMoeRemoteEvidence>()).get();
}
inline void bc_moe_remote_evidence_open(const void* ctx, int layers, int banks) {
    if (auto* t = bc_moe_remote_evidence_get(ctx, true)) t->reset(layers, banks);
}
inline void bc_moe_remote_evidence_issue(const void* ctx, int layer, int bank, int step,
                                         int64_t n_seq, int64_t n_decode, int draft_pass) {
    if (auto* t = bc_moe_remote_evidence_get(ctx))
        t->issue(layer, bank, step, BcMoeRemoteEvidence::phase(n_seq, n_decode, draft_pass));
}
inline void bc_moe_remote_evidence_flush(const void* ctx, int bank, int lo, int hi, int step) {
    if (auto* t = bc_moe_remote_evidence_get(ctx)) t->flush(bank, lo, hi, step);
}
inline bool bc_moe_remote_evidence_landed(const void* ctx, int layer, int bank, int step, bool ok) {
    if (auto* t = bc_moe_remote_evidence_get(ctx)) return t->landed(layer, bank, step, ok);
    return ok;
}
inline bool bc_moe_remote_evidence_credit(const void* ctx, int layer, int bank, int step, int passes) {
    if (auto* t = bc_moe_remote_evidence_get(ctx)) return t->credit(layer, bank, step, passes);
    return true;
}
/* Exit occurs after Ctx's pre-existing stream sync, never on the serving path.
 * Query each bank event once. This proves completion, NOT HeatEngine credit.
 * In particular, static plans without a consumer retain completed/pending rows. */
template<class Event>
inline void bc_moe_remote_evidence_exit(const void* ctx, int rank, uint64_t dropped,
                                         const Event* events) {
    auto* t = bc_moe_remote_evidence_get(ctx);
    if (!t) return;
    std::vector<int8_t> ready((size_t)t->banks, -1);
    for (int layer = 0; layer < t->layers; ++layer) {
        for (int bank = 0; bank < t->banks; ++bank) {
            auto* s = t->slot(layer, bank);
            if (!s || s->step < 0 || !s->flushed || s->dropped || s->completed) continue;
            if (ready[(size_t)bank] < 0)
                ready[(size_t)bank] = events[bank] && rad_event_query(events[bank]) == 1 ? 1 : 0;
            if (ready[(size_t)bank]) {
                s->completed = true;
                t->plus(t->count[s->phase].completed);
            }
        }
    }
    t->print(rank, dropped);
}
inline void bc_moe_remote_evidence_close(const void* ctx) {
    if (!bc_moe_remote_evidence_enabled()) return;
    std::lock_guard<std::mutex> lock(bc_moe_remote_evidence_mutex());
    bc_moe_remote_evidence_registry().erase(ctx);
}
/* Only when neither HeatEngine nor --expert-stats drains: use the *same* histogram
 * reader, with the same event query, rather than introduce a second producer. */
template<class Context>
bool bc_moe_remote_evidence_static_drain(Context& ctx, std::vector<int>& seen,
                                          int before_step, bool neither_consumer) {
    if (!neither_consumer || !bc_moe_remote_evidence_enabled()) return false;
    const int n = ctx.n_route_layers();
    if ((int)seen.size() != n) seen.assign((size_t)n, -1);
    for (int l = 0; l < n; ++l) {
        int after = seen[(size_t)l];
        for (;;) {
            const int bank = ctx.next_histogram(l, after, before_step);
            if (bank < 0) break;
            const int step = ctx.last_histogram_step(l, bank);
            if (!ctx.last_histogram(l, bank)) break;
            ctx.mark_credited(l, bank);
            after = step;
        }
        seen[(size_t)l] = after;
    }
    return true;
}
