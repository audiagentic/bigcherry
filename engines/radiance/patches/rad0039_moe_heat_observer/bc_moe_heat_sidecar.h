/* Issue-time metadata and optional static drain for the existing Ctx route banks.
 * The bank remains the sole histogram producer; this records no GPU-side counts.
 */
#pragma once
#include "rad_moe_heat.h"
#include <cstdint>
#include <limits>
#include <vector>

struct BcMoeHeatStamp {
    uint64_t generation = 0;
    uint32_t phase = RAD_MOE_HEAT_UNKNOWN;
    uint32_t flags = 0;
    uint32_t n_expert = 0;
    bool emitted = false;
};

inline uint32_t bc_moe_heat_phase(int n_seq, int n_decode, int draft_pass) {
    if (draft_pass != 0) return RAD_MOE_HEAT_DRAFT;
    if (n_seq <= 0 || n_decode < 0 || n_decode > n_seq) return RAD_MOE_HEAT_UNKNOWN;
    if (n_decode == 0) return RAD_MOE_HEAT_PREFILL;
    if (n_decode == n_seq) return RAD_MOE_HEAT_DECODE;
    return RAD_MOE_HEAT_MIXED;
}

/* A same-step replay replaces the bank's counts; it does NOT sum passes.
 * The exact number of reports is obtained from Ctx::last_histogram_reports.
 */
inline void bc_moe_heat_note_issue(BcMoeHeatStamp& stamp, uint64_t& generation,
                                   bool same_step, int n_seq, int n_decode,
                                   int draft_pass, int64_t n_expert, int64_t capacity) {
    const uint32_t phase = bc_moe_heat_phase(n_seq, n_decode, draft_pass);
    const bool invalid = n_expert <= 0 || n_expert != capacity ||
                         static_cast<uint64_t>(n_expert) > UINT32_MAX;
    if (!same_step) {
        if (generation != UINT64_MAX) ++generation;
        stamp.generation = generation == UINT64_MAX ? 0 : generation;
        stamp.emitted = false;
        stamp.phase = phase;
        stamp.flags = invalid ? static_cast<uint32_t>(RAD_MOE_HEAT_SOURCE_INCOMPLETE) : 0u;
        stamp.n_expert = n_expert > 0 && static_cast<uint64_t>(n_expert) <= UINT32_MAX
                             ? static_cast<uint32_t>(n_expert) : 0u;
    } else {
        if (stamp.phase != phase) {
            stamp.phase = RAD_MOE_HEAT_MIXED;
            stamp.flags |= RAD_MOE_HEAT_PASS_PHASE_MIXED;
        }
        if (invalid || stamp.n_expert != static_cast<uint64_t>(n_expert))
            stamp.flags |= RAD_MOE_HEAT_SOURCE_INCOMPLETE;
    }
}

/* Called before tiering_dispatch's optional RHP01 static drain. Return true only
 * when an observer actually drained the inactive heat/xstats path. The normal
 * heat and xstats paths continue to call the same Ctx::mark_credited method.
 * The original event-ready reader is still authoritative; no new GPU wait.
 */
bool bc_moe_heat_enabled() noexcept;
uint64_t bc_moe_heat_new_run() noexcept;
void bc_moe_heat_emit(const RadMoeHeatRowV1* row) noexcept;
template<class CtxType>
bool bc_moe_heat_static_drain(CtxType& ctx, std::vector<int>& seen,
                               int before_step, bool static_only,
                               uint64_t* credited) {
    if (!static_only || !bc_moe_heat_enabled()) return false;
    const int layers = ctx.n_route_layers();
    if (static_cast<int>(seen.size()) != layers) seen.assign(static_cast<size_t>(layers), -1);
    for (int l = 0; l < layers; ++l) {
        int after = seen[static_cast<size_t>(l)];
        for (;;) {
            const int bank = ctx.next_histogram(l, after, before_step);
            if (bank < 0 || !ctx.last_histogram(l, bank)) break;
            const int source_step = ctx.last_histogram_step(l, bank);
            if (source_step <= after) break;
            seen[static_cast<size_t>(l)] = after = source_step;
            ctx.mark_credited(l, bank);
            if (credited) ++*credited;
        }
    }
    return true;
}
