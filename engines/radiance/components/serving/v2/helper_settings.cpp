#include "helper_settings.h"

#include <algorithm>

namespace bc::serving::v2 {
namespace {
std::string join_formats(const std::vector<v2::ExpertFormat>& formats) {
    std::string out;
    for (v2::ExpertFormat f : formats) {
        if (!out.empty()) out += ",";
        out += std::string(v2::kExpertFormatNames[static_cast<size_t>(f)]);
    }
    return out;
}
template <class T>
uint32_t u32(T v) { return static_cast<uint32_t>(v); }
}  // namespace

v2::Result<HelperSettings> helper_settings(const v2::Manifest& m, std::string_view role) {
    auto view = v2::helper_view(m, role);
    if (!view) return view.error();
    const v2::HelperView& v = view.value();
    const v2::HelperCommon& c = v.settings;
    const auto& p = m.spec.processes;
    const v2::Handoff& h = *m.spec.handoff;   // a helper exists, so the handoff does

    HelperSettings s;
    s.role = std::string(role);
    s.content_sha256 = m.metadata.content_digest;
    s.model = m.spec.intent.model;
    s.plan_cache_key = v2::plan_cache_key(m);
    s.layout = m.spec.intent.layout;
    s.objective = std::string(v2::kObjectiveNames[static_cast<size_t>(m.spec.intent.service.objective)]);
    s.lab_profile = m.spec.intent.service.profile == v2::Profile::kLab;
    s.radiance_home = p.engine.runtime.radiance_home;
    s.helper = {u32(p.helpers.size()), u32(v.index), u32(p.engine.scheduler.tp), c.cpu.reference, c.runtime.power_save,
                c.runtime.queue_warm, c.combine.enabled, u32(c.combine.min_rows), u32(c.combine.threads),
                c.cpu.affinity == v2::Affinity::kAuto};
    s.handoff = {h.transport.shm,
                 h.placement.heat_file,
                 h.telemetry.histogram_file.value_or(""),
                 std::string(v2::kPlacementStrategyNames[static_cast<size_t>(h.placement.strategy)]),
                 u32(h.transport.max_rows),
                 u32(h.transport.small_rows),
                 u32(h.transport.gemv_large_rows),
                 u32(h.telemetry.period_s),
                 u32(h.kernels.moe_down_tile),
                 h.transport.skip_owned};
    s.plugin_file = v.process.plugin_file;
    s.formats = join_formats(v.process.formats);
    s.budget_mib = static_cast<uint64_t>(v.process.vram_budget_mib);
    for (const v2::HelperProcess& other : p.helpers) {
        s.budgets_mib.push_back(static_cast<uint64_t>(other.vram_budget_mib));
        s.shares.push_back(static_cast<uint64_t>(other.share));
    }
    s.balance = c.placement.balance;
    if (c.placement.balance_speeds_milli)
        for (int64_t milli : *c.placement.balance_speeds_milli) s.balance_speeds.push_back(static_cast<double>(milli) / 1000.0);
    else
        s.balance_speeds = {1.0, 1.0};
    for (int64_t lane : c.lanes.small_lanes) s.small_lanes.push_back(u32(lane));
    s.gemv = u32(c.gemv.mode);
    s.gemv_h2 = u32(c.gemv.h2);
    s.gemv_wide = c.gemv.wide;
    s.gemv_w4h2 = c.gemv.w4h2;
    s.gemv_mr = c.gemv.mr;
    s.rows_kernel = v.process.rows_kernel;
    s.rows = v.rows_active;
    s.rows_small = c.lanes.rows_small;
    s.fuseroute = c.kernels.fuseroute;
    s.vecio = c.kernels.vecio;
    s.sparse_act = c.kernels.sparse_act;
    s.lane_quant = c.kernels.lane_quant;
    s.moe_form = u32(c.kernels.moe_form);
    s.moe_long_rows = u32(c.kernels.moe_long_rows);
    s.native = c.kernels.native;
    s.fp16 = c.kernels.fp16;
    s.ahalf = c.kernels.ahalf;

    s.live_policy = h.placement.live_policy;
    if (m.spec.experiment && m.spec.experiment->placement) {
        const v2::ExperimentPlacement& e = *m.spec.experiment->placement;
        s.forced = e.forced;
        if (e.epoch_cycle_s) s.epoch_cycle_s = u32(*e.epoch_cycle_s);
    }
    if (m.spec.experiment && m.spec.experiment->diagnostics) {
        const v2::Diagnostics& d = *m.spec.experiment->diagnostics;
        if (d.gemv) {
            if (d.gemv->probe_mode) s.moe_probe = u32(*d.gemv->probe_mode);
            if (d.gemv->refusal_trace) s.refusal_trace = *d.gemv->refusal_trace;
        }
        if (d.lanes) {
            if (d.lanes->capture_file && *d.lanes->capture_file) s.capture = **d.lanes->capture_file;
            if (d.lanes->capture_every) s.capture_every = u32(*d.lanes->capture_every);
            if (d.lanes->capture_max) s.capture_max = u32(*d.lanes->capture_max);
            if (d.lanes->trace_level) s.trace = u32(*d.lanes->trace_level);
            if (d.lanes->test_stop_at && *d.lanes->test_stop_at) s.test_stop_at = **d.lanes->test_stop_at;
        }
        if (d.kernels) {
            if (d.kernels->plugin_override && *d.kernels->plugin_override) s.plugin = **d.kernels->plugin_override;
            if (d.kernels->poison) {
                switch (*d.kernels->poison) {
                    case v2::Poison::kOff: s.poison = "0"; break;
                    case v2::Poison::kZero: s.poison = "zero"; break;
                    case v2::Poison::kOne: s.poison = "one"; break;
                }
            }
            if (d.kernels->sweep_fast) s.sweep_fast = *d.kernels->sweep_fast;
        }
    }
    return s;
}

v2::Result<HelperSettings> load_helper_settings(const std::filesystem::path& path, std::string_view role, v2::PlanUse use) {
    auto m = v2::load_manifest(path, use);
    if (!m) return m.error();
    return helper_settings(m.value(), role);
}

}  // namespace bc::serving::v2
