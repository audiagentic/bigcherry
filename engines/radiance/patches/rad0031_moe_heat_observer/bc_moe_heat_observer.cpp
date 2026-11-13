/* Ctx-only bridge: the sidecar belongs to the existing route ring.
 * No second device collector, transfer, or per-layer stream synchronization.
 */
#include "ctx.h"
#include "bc_moe_heat_sidecar.h"
#include "rad_moe_heat.h"
#include <cstdint>
#include <limits>

uint64_t bc_moe_heat_new_run() noexcept;
void bc_moe_heat_emit(const RadMoeHeatRowV1* row) noexcept;

namespace rad {
void Ctx::heat_note_issue(RouteSlot& slot, int bank, bool again, int64_t n) {
    bc_moe_heat_note_issue(slot.heat[bank], route_generation_, again,
        batch_ ? batch_->n_seq : 0, batch_ ? batch_->n_seq_decode : 0,
        batch_ ? batch_->draft_pass : 0, n, route_n_expert_);
}

void Ctx::heat_credit(int layer, int bank) {
    RouteSlot& slot = route_[static_cast<size_t>(layer)];
    BcMoeHeatStamp& stamp = slot.heat[bank];
    if (stamp.emitted) return;
    stamp.emitted = true;
    if (!bc_moe_heat_enabled() || slot.step[bank] < 0) return;
    const int reps = slot.reports[bank];
    const int32_t* const counts = slot.staging[bank];
    if (!counts || reps <= 0 || static_cast<uint64_t>(reps) > UINT32_MAX ||
        rank_ < 0 || layer < 0 || stamp.generation == 0 || run_generation_ == 0)
        return;
    RadMoeHeatRowV1 row{};
    row.abi_version = RAD_MOE_HEAT_ABI_V1;
    row.struct_size = sizeof row;
    row.run_generation = run_generation_;
    row.engine_step = static_cast<uint64_t>(slot.step[bank]);
    row.bank_generation = stamp.generation;
    row.dropped_source_total = route_dropped_;
    row.rank = static_cast<uint32_t>(rank_);
    row.layer = static_cast<uint32_t>(layer);
    row.expert_base = 0;  /* Ctx route histogram uses model-global expert indexes */
    row.n_expert = stamp.n_expert;
    row.pass_reports = static_cast<uint32_t>(reps);
    row.phase = stamp.phase;
    row.flags = stamp.flags | RAD_MOE_HEAT_EVENT_READY;
    row.counts = counts;
    bc_moe_heat_emit(&row);
}
} // namespace rad
