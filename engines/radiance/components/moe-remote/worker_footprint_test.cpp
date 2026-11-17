// RAD49: the helper's own device memory besides its expert arena (worker_footprint.h), and the expert budget derived from it for
// a layout without budget-mib. The numbers are the flashnext-tp2xtx measuring start on the 16 GiB RX 6900 XT tail helper
// (runs/rad49-hw2): with the budget derived as the whole card less its reserve, the arena plus the helper's lanes overflowed by
// 85,007,248 bytes; with the footprint taken out of the budget it fits.
#include "worker_footprint.h"
#include "worker_slot_pool.h"

#include <cstdio>
#include <cstdlib>

namespace fp = r3::worker_footprint;

/* This is the same ensure_lane_buffers template the HIP worker instantiates, with a CPU allocator.
 * Every requested buffer has its own counter, so both omitted and oversized fields fail the footprint test. */
struct CpuBuffer {
    uint64_t bytes = 0;
    bool ensure(uint64_t n) { if (n > bytes) bytes = n; return true; }
};
struct CpuLane {
    CpuBuffer d_counts, d_rowmask, d_cursor, d_act, d_flat, d_sorted, d_off, d_ew, d_inv;
    CpuBuffer d_aq, d_as, d_gu, d_mid, d_mid_s, d_down, d_dbg, d_stats, d_publish_limit, d_chunk_cnt;
    std::array<CpuBuffer, 2> gu_code, gu_scale, dn_code, dn_scale;
    uint64_t allocated() const {
        const CpuBuffer* buffers[] = {&d_counts, &d_rowmask, &d_cursor, &d_act, &d_flat, &d_sorted,
            &d_off, &d_ew, &d_inv, &d_aq, &d_as, &d_gu, &d_mid, &d_mid_s, &d_down,
            &d_dbg, &d_stats, &d_publish_limit, &d_chunk_cnt};
        uint64_t n = 0;
        for (const CpuBuffer* b : buffers) n += b->bytes;
        for (unsigned parity = 0; parity < 2; ++parity)
            n += gu_code[parity].bytes + gu_scale[parity].bytes + dn_code[parity].bytes + dn_scale[parity].bytes;
        return n;
    }
};


static void check(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "r3_worker_footprint_test: FAILED %s\n", why); std::exit(1); }
}

int main() {
    using namespace r3::moe_shm;
    constexpr uint64_t MiB = 1ull << 20;
    /* Flash-Next: 48 layers and the MTP layer, 512 experts of width 640, hidden 2560, top-10; one small class of 8 rows and the
     * 4096-row large lane (the schema defaults), power save on. The measured slot stride is 2,534,400 bytes. */
    const fp::Shape flash{49, 512, 2560, 640, 10, 128};
    constexpr uint64_t kStride = 2534400;
    fp::Lanes lanes{};
    lanes.small_rows[0] = 8; lanes.small_count = 1; lanes.large_rows = 4096;

    /* helper 1 of 2 behind TP2 serves slots 2 and 3: two small and two large lanes */
    lanes.open = fp::served_lanes(2, 1, 2, 1);
    uint32_t open = 0;
    for (uint32_t lane = 0; lane < lanes.open.size(); ++lane) {
        if (!lanes.open[lane]) continue;
        ++open;
        check(slot_helper(lane % slots_for(2, 2), 2) == 1, "helper 1's lanes are its own slots");
    }
    check(open == 4, "two consumer ranks x (one small class + the large lane)");
    /* the hardware log: lanes 959,942,096 (of which 328,320 the plugins' scratch, not in the footprint), pointer banks 1,612,688,
     * slot map 50,176, incoming 2,534,400 */
    constexpr uint64_t kMeasuredLanes = 959942096, kMeasuredBanks = 1612688, kMeasuredMap = 50176;
    const uint64_t lane_bytes = fp::lanes_bytes(flash, lanes);
    check(lane_bytes <= kMeasuredLanes && kMeasuredLanes - lane_bytes < 1 * MiB,
          "the footprint's lanes are the measured lanes less the plugin scratch (under 1 MiB of it)");
    check(fp::pointer_bank_bytes(flash) == kMeasuredBanks, "pointer banks as measured");
    check(fp::slot_map_bytes(flash) == kMeasuredMap, "slot map as measured");

    /* the 16 GiB card (16368 MiB, archs.toml gfx1030 reserve-mib 1024) with its measured free memory */
    constexpr uint64_t kCard = 16368 * MiB, kReserve = 1024 * MiB;
    /* free before the lanes, and free at the arena (the lanes and the lane streams' queues and code taken) */
    constexpr uint64_t kFreeBeforeLanes = 17131634688ull, free_now = 16007561216ull;
    auto required = [&](uint64_t budget) {
        const auto plan = r3::worker_slots::budget_plan(budget, kStride, uint64_t(flash.layers) * flash.experts);
        const r3::worker_slots::DeviceMemory m{kMeasuredLanes, kMeasuredBanks, kMeasuredMap, kStride, plan.expert_bytes};
        return r3::worker_slots::check_device_memory(m, kFreeBeforeLanes, free_now);
    };
    /* the old derivation: the whole card less the reserve as the EXPERT budget -- the exact shortfall of the hardware start */
    const auto old_fit = required(kCard - kReserve);
    check(!old_fit.fits && old_fit.shortfall == 85007248, "the old derivation reproduces the measured shortfall");
    /* the derivation now: less the helper's own lanes, banks, slot map and incoming slot */
    const uint64_t fixed = fp::fixed_bytes(flash, lanes, kStride);
    const uint64_t budget = fp::expert_budget_bytes(kCard, kReserve, fixed);
    check(budget > 0 && budget < kCard - kReserve && budget + fixed == kCard - kReserve, "budget = card - reserve - footprint");
    const uint64_t budget_mib = budget / MiB;   /* what bc-serve writes (whole MiB, rounded down) */
    const auto fit = required(budget_mib * MiB);
    check(fit.fits, "the derived budget's arena and the helper's own memory fit the card's measured free VRAM");
    const uint64_t slack = free_now - fit.future;
    check(slack < kReserve, "and what is left is within the arch reserve (context, queues, code objects, plugin scratch)");
    std::printf("gfx1030 tail: derived %llu MiB (was %llu), %llu experts, %llu MiB unused of the free VRAM\n",
                (unsigned long long)budget_mib, (unsigned long long)((kCard - kReserve) / MiB),
                (unsigned long long)(budget_mib * MiB / kStride), (unsigned long long)(slack / MiB));

    /* the stride bound covers the worker's per-plane rounding */
    check(fp::stride_bound(kStride) >= kStride && fp::stride_bound(kStride) - kStride == 4 * 255, "stride bound");
    /* power save off opens every protocol lane: a larger footprint, a smaller budget */
    fp::Lanes all = lanes;
    all.open = fp::served_lanes(2, 1, 2, 1, false);
    check(fp::lanes_bytes(flash, all) == 2 * lane_bytes, "power save off: all four slots' lanes");
    /* a card too small for the helper's own memory yields no budget */
    check(fp::expert_budget_bytes(kReserve + fixed, kReserve, fixed) == 0 && fp::expert_budget_bytes(kReserve, kReserve, 0) == 0,
          "no budget when the footprint takes the card");
    /* a one-rank engine: helper k serves its one slot */
    const auto one = fp::served_lanes(3, 2, 1, 1);
    for (uint32_t lane = 0; lane < one.size(); ++lane)
        check(one[lane] == (lane % slots_for(3,1) == 2 &&
              (!lane_is_small(lane, slots_for(3,1)) || lane_class(lane, slots_for(3,1)) < 1)), "TP1 helper lanes");
    /* PR #771 review: the lanes come from the helper's lane settings, the same functions for the worker (worker_shape.h
     * helper_small_classes, lanes_open) and for bc-serve: an empty small-lane list is one class of small_rows (within the large
     * lane), a list is its classes; both power-save modes. */
    {
        const auto one = fp::small_classes({}, 8, 4096);
        check(one.count == 1 && one.rows[0] == 8, "empty list: one class of small_rows");
        check(fp::small_classes({}, 64, 32).rows[0] == 32, "small_rows within the large lane");
        const auto four = fp::small_classes({4, 8, 16, 24}, 8, 4096);
        check(four.count == 4 && four.rows[0] == 4 && four.rows[3] == 24, "a list of four classes");
        for (const bool save : {true, false}) {
            const fp::Lanes l1 = fp::lanes_for(one, 4096, 2, 1, 2, save), l4 = fp::lanes_for(four, 4096, 2, 1, 2, save);
            uint32_t n1 = 0, n4 = 0;
            for (uint32_t lane = 0; lane < l1.open.size(); ++lane) { n1 += l1.open[lane]; n4 += l4.open[lane]; }
            const uint32_t slots = save ? 2u : slots_for(2,2);   /* power save: helper 1's two slots; off: every protocol slot */
            check(n1 == slots * 2 && n4 == slots * 5, "open lanes: (classes + the large lane) a served slot");
            uint64_t want = 0;
            for (uint32_t lane = 0; lane < l4.open.size(); ++lane) {
                if (!l4.open[lane]) continue;
                const uint32_t cls = lane_class(lane, slots_for(2,2));
                const uint64_t cap = cls < kSmallClasses ? four.rows[cls] : 4096;
                want += fp::lane_bytes(flash, cap, cap * flash.top_k, cls == kSmallClasses);
            }
            check(fp::lanes_bytes(flash, l4) == want && want > fp::lanes_bytes(flash, l1), "four classes cost their own lanes");
            /* the worker with power save off opens every lane (lanes_open: all true); a disabled class gets no buffers, so the
             * footprint of that mask is the planner's */
            fp::Lanes all = l4;
            std::fill(all.open.begin(), all.open.end(), true);
            check(fp::lanes_bytes(flash, all) == (save ? fp::lanes_bytes(flash, fp::lanes_for(four, 4096, 2, 1, 2, false))
                                                       : fp::lanes_bytes(flash, l4)),
                  "the worker's all-lanes mask costs what the planner's power-save-off plan does");
            fp::Lanes all1 = l1;
            std::fill(all1.open.begin(), all1.open.end(), true);
            check(fp::lanes_bytes(flash, all1) == fp::lanes_bytes(flash, fp::lanes_for(one, 4096, 2, 1, 2, false)),
                  "and so for one class: classes past the count are not prepared");
        }
    }
    /* The real lane-buffer preparation, running here with CPU-only stub ensure() methods.
     * Audit every per-lane GPU allocation, including the eight pointer tables and small/large differences.
     * Protocol 4096 and configured 64 must not reserve different chunk counts in the worker and the planner. */
    for (const uint32_t cap : {1u, 8u, 64u, 256u, 257u, 4096u}) {
        const uint64_t routes = uint64_t(cap) * flash.top_k;
        for (const bool large : {false, true}) {
            CpuLane w;
            check(fp::ensure_lane_buffers(w, flash, cap, routes, routes, large), "CPU lane preparation");
            const uint64_t E = flash.experts, H = flash.hidden, I = flash.intermediate, G = flash.act_group;
            check(w.d_counts.bytes == E * 4 && w.d_cursor.bytes == E * 4 && w.d_rowmask.bytes == cap,
                  "counts, cursor, mask allocations");
            check(w.d_act.bytes == cap * H * 2 && w.d_flat.bytes == routes * 4 &&
                  w.d_sorted.bytes == routes * 4 && w.d_off.bytes == (E + 1) * 4,
                  "activations and routing allocations");
            check(w.d_ew.bytes == routes * 2 && w.d_inv.bytes == routes * 4 &&
                  w.d_aq.bytes == cap * H && w.d_as.bytes == cap * (H / G) * 4,
                  "route weights and quantised activations");
            check(w.d_gu.bytes == routes * 2 * I * 2 && w.d_mid.bytes == routes * I &&
                  w.d_mid_s.bytes == routes * (I / G) * 4 && w.d_down.bytes == routes * H * 2 &&
                  w.d_dbg.bytes == routes * 2 * I * 2, "gate, middle, down and debug allocations");
            check(w.d_stats.bytes == 3 * sizeof(uint64_t) && w.d_publish_limit.bytes == sizeof(uint32_t),
                  "lane stats and publish limit allocations");
            const uint64_t chunks = large ? ((uint64_t(cap) + 255) / 256) * sizeof(uint32_t) : 0;
            check(w.d_chunk_cnt.bytes == chunks && fp::chunk_counter_bytes(cap) == ((uint64_t(cap) + 255) / 256) * 4,
                  "configured-capacity chunk counters");
            for (unsigned parity = 0; parity < 2; ++parity)
                check(w.gu_code[parity].bytes == (E / 2) * sizeof(void*) &&
                      w.gu_scale[parity].bytes == (E / 2) * sizeof(void*) &&
                      w.dn_code[parity].bytes == (E / 2) * sizeof(void*) &&
                      w.dn_scale[parity].bytes == (E / 2) * sizeof(void*), "all eight parity pointer tables");
            check(w.allocated() == fp::lane_bytes(flash, cap, routes, large),
                  "CPU-prepared buffers equal the planner's lane footprint");
        }
    }
    check(fp::chunk_counter_bytes(64) == 4 && fp::chunk_counter_bytes(4096) == 64,
          "64-row versus protocol-4096 counters");
    check(2 * (fp::chunk_counter_bytes(4096) - fp::chunk_counter_bytes(64)) == 120,
          "run-6 mismatch reproduced before the fix: two large lanes, 120 bytes");
    std::puts("r3_worker_footprint_test: ok");
    return 0;
}
