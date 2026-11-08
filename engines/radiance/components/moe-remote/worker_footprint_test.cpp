// RAD49: the helper's own device memory besides its expert arena (worker_footprint.h), and the expert budget derived from it for
// a layout without budget-mib. The numbers are the flashnext-tp2xtx measuring start on the 16 GiB RX 6900 XT tail helper
// (runs/rad49-hw2): with the budget derived as the whole card less its reserve, the arena plus the helper's lanes overflowed by
// 85,007,248 bytes; with the footprint taken out of the budget it fits.
#include "worker_footprint.h"
#include "worker_slot_pool.h"

#include <cstdio>
#include <cstdlib>

namespace fp = r3::worker_footprint;

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
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        if (!lanes.open[lane]) continue;
        ++open;
        check(slot_helper(lane % kMaxSlots, 2) == 1, "helper 1's lanes are its own slots");
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
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        check(one[lane] == (lane % kMaxSlots == 2 && (!lane_is_small(lane) || lane_class(lane) < 1)), "TP1 helper lanes");
    std::puts("r3_worker_footprint_test: ok");
    return 0;
}
