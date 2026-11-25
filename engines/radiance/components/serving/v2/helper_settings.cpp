#include "helper_settings.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <regex>
#include <set>
#include <type_traits>

#include "document_bytes.h"
#include "serving_codec.h"
#include "strict_json.h"
#include "wire_reader.h"

namespace bc::serving {
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
    s.live_calibration = h.placement.live_calibration;
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

namespace {
using Json = nlohmann::json;
using wire::Fault;
using wire::Wire;

template <class T>
std::optional<T> nested(Wire& w, const char* key) {
    const Json& v = w.at(key);
    if (v.is_null()) return std::nullopt;
    T out{};
    if (const auto e = v2::decode(v, w.path(key), &out)) throw Fault{e->code, e->path, e->message};
    return out;
}
}  // namespace

Json helper_settings_to_json(const HelperSettings& s) {
    Json o;
    o["helper"] = {{"count", s.helper.count}, {"index", s.helper.index}, {"engine_ranks", s.helper.engine_ranks},
                   {"cpu", s.helper.cpu}, {"power_save", s.helper.power_save}, {"queue_warm", s.helper.queue_warm},
                   {"combine", s.helper.combine}, {"combine_min_rows", s.helper.combine_min_rows},
                   {"combine_threads", s.helper.combine_threads}, {"cpu_set_auto", s.helper.cpu_set_auto}};
    o["handoff"] = {{"shm", s.handoff.shm}, {"heat", s.handoff.heat}, {"histogram", s.handoff.histogram},
                    {"placement", s.handoff.placement}, {"max_rows", s.handoff.max_rows},
                    {"small_rows", s.handoff.small_rows}, {"gemv_large_rows", s.handoff.gemv_large_rows},
                    {"histogram_period_s", s.handoff.histogram_period_s}, {"moe_down_tile", s.handoff.moe_down_tile},
                    {"skip_owned", s.handoff.skip_owned}};
    o["role"] = s.role;
    o["model"] = s.model;
    o["radiance_home"] = s.radiance_home;
    o["measuring"] = s.measuring;
    o["plan_cache_key"] = s.plan_cache_key;
    o["layout"] = s.layout;
    o["objective"] = s.objective;
    o["lab_profile"] = s.lab_profile;
    o["plugin"] = s.plugin;
    o["plugin_file"] = s.plugin_file;
    o["formats"] = s.formats;
    o["budgets_mib"] = s.budgets_mib;
    o["shares"] = s.shares;
    // The wire is integer-only like every v2 document: speeds travel in thousandths, as the manifest holds them.
    Json speeds = Json::array();
    for (double v : s.balance_speeds) speeds.push_back(static_cast<uint64_t>(std::llround(v * 1000.0)));
    o["balance_speeds_milli"] = speeds;
    o["small_lanes"] = s.small_lanes;
    o["budget_mib"] = s.budget_mib;
    o["gemv"] = s.gemv;
    o["gemv_h2"] = s.gemv_h2;
    o["moe_form"] = s.moe_form;
    o["moe_long_rows"] = s.moe_long_rows;
    o["gemv_wide"] = s.gemv_wide;
    o["gemv_w4h2"] = s.gemv_w4h2;
    o["gemv_mr"] = s.gemv_mr;
    o["rows"] = s.rows;
    o["rows_small"] = s.rows_small;
    o["rows_kernel"] = s.rows_kernel;
    o["fuseroute"] = s.fuseroute;
    o["vecio"] = s.vecio;
    o["sparse_act"] = s.sparse_act;
    o["lane_quant"] = s.lane_quant;
    o["native"] = s.native;
    o["fp16"] = s.fp16;
    o["ahalf"] = s.ahalf;
    o["balance"] = s.balance;
    o["capture"] = s.capture;
    o["poison"] = s.poison;
    o["test_stop_at"] = s.test_stop_at;
    o["capture_every"] = s.capture_every;
    o["capture_max"] = s.capture_max;
    o["trace"] = s.trace;
    o["moe_probe"] = s.moe_probe;
    o["refusal_trace"] = s.refusal_trace;
    o["sweep_fast"] = s.sweep_fast;
    o["live_policy"] = s.live_policy ? v2::encode(*s.live_policy) : Json(nullptr);
    o["live_calibration"] = s.live_calibration ? v2::encode(*s.live_calibration) : Json(nullptr);
    o["forced"] = s.forced ? v2::encode(*s.forced) : Json(nullptr);
    o["epoch_cycle_s"] = s.epoch_cycle_s;
    return o;
}

v2::Result<HelperSettings> helper_settings_from_json(const Json& j) {
    try {
        Wire w(j, "helper_settings");
        HelperSettings s;
        {
            Wire h(w.at("helper"), "helper_settings.helper");
            s.helper.count = h.u32("count");
            s.helper.index = h.u32("index");
            s.helper.engine_ranks = h.u32("engine_ranks");
            s.helper.cpu = h.boolean("cpu");
            s.helper.power_save = h.boolean("power_save");
            s.helper.queue_warm = h.boolean("queue_warm");
            s.helper.combine = h.boolean("combine");
            s.helper.combine_min_rows = h.u32("combine_min_rows");
            s.helper.combine_threads = h.u32("combine_threads");
            s.helper.cpu_set_auto = h.boolean("cpu_set_auto");
            h.done();
        }
        {
            Wire h(w.at("handoff"), "helper_settings.handoff");
            s.handoff.shm = h.text("shm");
            s.handoff.heat = h.text("heat");
            s.handoff.histogram = h.text("histogram");
            s.handoff.placement = h.text("placement");
            s.handoff.max_rows = h.u32("max_rows");
            s.handoff.small_rows = h.u32("small_rows");
            s.handoff.gemv_large_rows = h.u32("gemv_large_rows");
            s.handoff.histogram_period_s = h.u32("histogram_period_s");
            s.handoff.moe_down_tile = h.u32("moe_down_tile");
            s.handoff.skip_owned = h.boolean("skip_owned");
            h.done();
        }
        s.role = w.text("role");
        s.model = w.text("model");
        s.radiance_home = w.text("radiance_home");
        s.measuring = w.boolean("measuring");
        s.plan_cache_key = w.text("plan_cache_key");
        s.layout = w.text("layout");
        s.objective = w.text("objective");
        s.lab_profile = w.boolean("lab_profile");
        s.plugin = w.text("plugin");
        s.plugin_file = w.text("plugin_file");
        s.formats = w.text("formats");
        s.budgets_mib = w.list<uint64_t>("budgets_mib");
        s.shares = w.list<uint64_t>("shares");
        for (uint64_t milli : w.list<uint64_t>("balance_speeds_milli")) s.balance_speeds.push_back(static_cast<double>(milli) / 1000.0);
        s.small_lanes = w.list<uint32_t>("small_lanes");
        s.budget_mib = w.u64("budget_mib");
        s.gemv = w.u32("gemv");
        s.gemv_h2 = w.u32("gemv_h2");
        s.moe_form = w.u32("moe_form");
        s.moe_long_rows = w.u32("moe_long_rows");
        s.gemv_wide = w.boolean("gemv_wide");
        s.gemv_w4h2 = w.boolean("gemv_w4h2");
        s.gemv_mr = w.boolean("gemv_mr");
        s.rows = w.boolean("rows");
        s.rows_small = w.boolean("rows_small");
        s.rows_kernel = w.boolean("rows_kernel");
        s.fuseroute = w.boolean("fuseroute");
        s.vecio = w.boolean("vecio");
        s.sparse_act = w.boolean("sparse_act");
        s.lane_quant = w.boolean("lane_quant");
        s.native = w.boolean("native");
        s.fp16 = w.boolean("fp16");
        s.ahalf = w.boolean("ahalf");
        s.balance = w.boolean("balance");
        s.capture = w.text("capture");
        s.poison = w.text("poison");
        s.test_stop_at = w.text("test_stop_at");
        s.capture_every = w.u32("capture_every");
        s.capture_max = w.u32("capture_max");
        s.trace = w.u32("trace");
        s.moe_probe = w.u32("moe_probe");
        s.refusal_trace = w.boolean("refusal_trace");
        s.sweep_fast = w.boolean("sweep_fast");
        s.live_policy = nested<v2::LivePolicy>(w, "live_policy");
        s.live_calibration = nested<v2::LiveCalibration>(w, "live_calibration");
        s.forced = nested<v2::ForcedChurn>(w, "forced");
        s.epoch_cycle_s = w.u32("epoch_cycle_s");
        w.done();
        // What the owner promised and the process was started as must agree before anything touches a device.
        static const std::regex role_form("^helper[0-9]+$");
        if (!std::regex_match(s.role, role_form))
            throw Fault{v2::ConfigErrorCode::invalid_value, "helper_settings.role", "not a helper role: " + s.role};
        if (s.helper.count == 0 || s.helper.index >= s.helper.count || s.role != "helper" + std::to_string(s.helper.index))
            throw Fault{v2::ConfigErrorCode::ownership_mismatch, "helper_settings.helper",
                        "index " + std::to_string(s.helper.index) + " of " + std::to_string(s.helper.count) +
                            " helpers does not match role " + s.role};
        if (s.budgets_mib.size() != s.helper.count || s.shares.size() != s.helper.count ||
            s.budget_mib != s.budgets_mib[s.helper.index])
            throw Fault{v2::ConfigErrorCode::ownership_mismatch, "helper_settings.budgets_mib",
                        "the budgets and shares do not describe this helper's place among " +
                            std::to_string(s.helper.count)};
        if (s.handoff.shm.size() < 2 || s.handoff.shm.front() != '/')
            throw Fault{v2::ConfigErrorCode::invalid_value, "helper_settings.handoff.shm", "not a shared-memory segment name"};
        return s;
    } catch (const Fault& f) {
        return v2::make_error(f.code, f.path, f.message);
    }
}

v2::Result<HelperSettings> read_helper_settings(int fd) {
    std::string bytes;
    try {
        bytes = v2::read_sealed_document(fd, "helper settings descriptor");
    } catch (const std::exception& e) {
        return v2::make_error(v2::ConfigErrorCode::io_failure, "helper_settings", e.what());
    }
    auto parsed = v2::parse_strict_json(bytes, "helper settings descriptor");
    if (!parsed) return parsed.error();
    return helper_settings_from_json(parsed.value());
}

v2::Result<HelperSettings> load_helper_settings(const std::filesystem::path& path, std::string_view role, v2::PlanUse use) {
    auto m = v2::load_manifest(path, use);
    if (!m) return m.error();
    return helper_settings(m.value(), role);
}

}  // namespace bc::serving
