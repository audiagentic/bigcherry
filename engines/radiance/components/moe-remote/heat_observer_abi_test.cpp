/* Card-free ABI and fake-Ctx source oracle for rad0031 (RHP02 slice A). */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../../patches/rad0031_moe_heat_observer/rad_moe_heat.h"
#include "../../patches/rad0031_moe_heat_observer/bc_moe_heat_sidecar.h"
#include <atomic>
#include <cassert>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

struct Sample {
    uint64_t step;
    uint64_t generation;
    uint32_t phase;
    uint32_t weight;
    uint32_t flags;
    std::vector<int32_t> counts;
};
struct Sink {
    std::vector<Sample> samples;
};
static int take(void* opaque, const RadMoeHeatRowV1* row) {
    auto& sink = *static_cast<Sink*>(opaque);
    assert(row->abi_version == RAD_MOE_HEAT_ABI_V1 && row->struct_size == sizeof *row);
    assert(row->reserved == 0 && (row->flags & RAD_MOE_HEAT_EVENT_READY));
    sink.samples.push_back({row->engine_step, row->bank_generation, row->phase,
                            row->pass_reports, row->flags,
                            std::vector<int32_t>(row->counts, row->counts + row->n_expert)});
    return 0;
}
static int reject(void*, const RadMoeHeatRowV1*) { return -1; }
static std::atomic<bool> entered{false}, released{false};
static int stalled(void*, const RadMoeHeatRowV1*) {
    entered.store(true);
    while (!released.load()) std::this_thread::yield();
    return 0;
}

static RadMoeHeatRowV1 make_row(const int32_t* counts, uint64_t step,
                               const BcMoeHeatStamp& stamp, int reports) {
    RadMoeHeatRowV1 row{};
    row.abi_version = RAD_MOE_HEAT_ABI_V1;
    row.struct_size = sizeof row;
    row.run_generation = 1;
    row.engine_step = step;
    row.bank_generation = stamp.generation;
    row.rank = 0;
    row.layer = 1;
    row.n_expert = stamp.n_expert;
    row.pass_reports = static_cast<uint32_t>(reports);
    row.phase = stamp.phase;
    row.flags = stamp.flags | RAD_MOE_HEAT_EVENT_READY;
    row.counts = counts;
    return row;
}

/* Fake credit uses the package's issue stamp and one-credit-per-generation
 * rule; same-step draft reports overwrite, not sum. No GPU or mock DMA. */
struct FakeCtx {
    struct Bank { int step; int reports; bool ready, credited; BcMoeHeatStamp stamp; int32_t h[3]; };
    std::vector<Bank> banks;
    uint64_t generation = 0;
    int credits = 0;
    int n_route_layers() const { return 1; }
    void issue(int step, int reports, int phase_kind, bool ready, int a) {
        BcMoeHeatStamp stamp{};
        bc_moe_heat_note_issue(stamp, generation, false, 1,
                              phase_kind == RAD_MOE_HEAT_PREFILL ? 0 : 1,
                              phase_kind == RAD_MOE_HEAT_DRAFT ? -1 : 0, 3, 3);
        banks.push_back({step, reports, ready, false, stamp, {a, a+1, a+2}});
    }
    int next_histogram(int, int after, int before) const {
        for (size_t i = 0; i < banks.size(); ++i)
            if (banks[i].step > after && banks[i].step < before &&
                banks[i].ready && !banks[i].credited)
                return static_cast<int>(i);
        return -1;
    }
    const int32_t* last_histogram(int, int bank) const { return banks[bank].h; }
    int last_histogram_step(int, int bank) const { return banks[bank].step; }
    void mark_credited(int, int bank) {
        auto& b = banks[bank];
        b.credited = true;
        ++credits;
        if (b.stamp.emitted) return;
        b.stamp.emitted = true;
        RadMoeHeatRowV1 row = make_row(b.h, b.step, b.stamp, b.reports);
        bc_moe_heat_emit(&row);
    }
};

int main() {
    static_assert(sizeof(RadMoeHeatRowV1) % alignof(uint64_t) == 0);
    assert(rad_moe_heat_register_v1(999, sizeof(RadMoeHeatRowV1), nullptr, nullptr)
           == RAD_MOE_HEAT_REGISTER_BAD_ABI);
    assert(rad_moe_heat_register_v1(RAD_MOE_HEAT_ABI_V1,
           sizeof(RadMoeHeatRowV1)-1, nullptr, nullptr)
           == RAD_MOE_HEAT_REGISTER_BAD_ABI);
    assert(rad_moe_heat_register_v1(RAD_MOE_HEAT_ABI_V1,
           sizeof(RadMoeHeatRowV1), nullptr, reinterpret_cast<void*>(1))
           == RAD_MOE_HEAT_REGISTER_BAD_ARGUMENT);
    assert(!bc_moe_heat_enabled());
    Sink sink;
    BcMoeHeatStamp meta{};
    uint64_t generation = 0;
    bc_moe_heat_note_issue(meta, generation, false, 6, 0, 0, 3, 3);
    assert(meta.phase == RAD_MOE_HEAT_PREFILL && generation == 1);
    bc_moe_heat_note_issue(meta, generation, true, 6, 0, -1, 3, 3);
    assert(meta.phase == RAD_MOE_HEAT_MIXED &&
           (meta.flags & RAD_MOE_HEAT_PASS_PHASE_MIXED) && generation == 1);
    int32_t counts[3] = {4, 5, 6};
    RadMoeHeatRowV1 row = make_row(counts, 10, meta, 4);
    bc_moe_heat_emit(&row); /* no observer, no side effect */
    assert(sink.samples.empty());
    assert(rad_moe_heat_register_v1(RAD_MOE_HEAT_ABI_V1, sizeof row, take, &sink) == 0);
    assert(rad_moe_heat_register_v1(RAD_MOE_HEAT_ABI_V1, sizeof row, take, &sink)
           == RAD_MOE_HEAT_REGISTER_OCCUPIED);
    bc_moe_heat_emit(&row);
    assert(sink.samples.size() == 1 && sink.samples[0].weight == 4 &&
           sink.samples[0].counts[2] == 6 &&
           sink.samples[0].phase == RAD_MOE_HEAT_MIXED);
    counts[2] = 99;
    assert(sink.samples[0].counts[2] == 6); /* callback made a private copy */
    uint64_t failures = rad_moe_heat_callback_errors_v1();
    row.struct_size -= 1;
    bc_moe_heat_emit(&row);
    row.struct_size += 1;
    assert(rad_moe_heat_callback_errors_v1() == failures + 1);
    assert(sink.samples.size() == 1);

    /* Ready source generation; delayed bank, source step order, static drain,
     * decode and MTP weight. Unready rows are never observed. */
    FakeCtx ctx;
    ctx.issue(2, 1, RAD_MOE_HEAT_PREFILL, true, 8);
    ctx.issue(4, 4, RAD_MOE_HEAT_DRAFT, false, 12);
    ctx.issue(6, 1, RAD_MOE_HEAT_DECODE, true, 20);
    std::vector<int> seen;
    uint64_t credited = 0;
    assert(!bc_moe_heat_static_drain(ctx, seen, 7, false, &credited));
    assert(bc_moe_heat_static_drain(ctx, seen, 7, true, &credited));
    assert(credited == 2 && ctx.credits == 2 && sink.samples.size() == 3);
    assert(sink.samples[1].step == 2 && sink.samples[2].step == 6);
    ctx.banks[1].ready = true;
    assert(bc_moe_heat_static_drain(ctx, seen, 7, true, &credited));
    assert(credited == 2); /* late step 4 is not replayed out of source order */
    ctx.mark_credited(0, 0); /* repeated mark: one callback per generation */
    assert(sink.samples.size() == 3);
    bc_moe_heat_note_issue(meta, generation, false, 6, 2, 0, 2, 3);
    assert((meta.flags & RAD_MOE_HEAT_SOURCE_INCOMPLETE) &&
           meta.phase == RAD_MOE_HEAT_MIXED);
    assert(bc_moe_heat_phase(6, 7, 0) == RAD_MOE_HEAT_UNKNOWN);

    assert(rad_moe_heat_register_v1(RAD_MOE_HEAT_ABI_V1, sizeof row, nullptr, nullptr) == 0);
    assert(!bc_moe_heat_enabled());
    assert(!bc_moe_heat_static_drain(ctx, seen, 8, true, &credited));
    assert(rad_moe_heat_register_v1(RAD_MOE_HEAT_ABI_V1, sizeof row, reject, nullptr) == 0);
    failures = rad_moe_heat_callback_errors_v1();
    bc_moe_heat_emit(&row);
    assert(rad_moe_heat_callback_errors_v1() == failures+1);
    assert(rad_moe_heat_register_v1(RAD_MOE_HEAT_ABI_V1, sizeof row, nullptr, nullptr) == 0);

    /* Early unregister waits for a callback already using borrowed user memory. */
    assert(rad_moe_heat_register_v1(RAD_MOE_HEAT_ABI_V1, sizeof row, stalled, nullptr) == 0);
    std::thread producer([&] { bc_moe_heat_emit(&row); });
    while (!entered.load()) std::this_thread::yield();
    std::thread unreg([&] {
        assert(rad_moe_heat_register_v1(RAD_MOE_HEAT_ABI_V1, sizeof row, nullptr, nullptr) == 0);
    });
    released.store(true);
    producer.join();
    unreg.join();
    assert(!bc_moe_heat_enabled());
    bc_moe_heat_emit(&row);
    return 0;
}
