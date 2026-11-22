#include "live_policy.h"
#include "manifest_writer.h"
#include "experts.h"
#include "plan_cache.h"

#include <algorithm>
#include <exception>
#include <unistd.h>
#include <limits>
#include <stdexcept>

namespace bc::serving {
namespace {
// The placement planner (worker_heat.h, worker_share::plan_tail) takes the LAST helper to be the tail and every other helper a head.
// Another arrangement would make it treat a head as a tail (flashnext-2xtx has no tail) or override a weighted head split
// (flashnext-2xtx-rdna2-head); a layout it was not built for gets no policy, with the reason, until the planner is topology-aware (RHP20).
bool planner_arrangement_supported(const ManifestInputs& in, std::string* skipped_reason) {
    if (in.resolved.manifest.is_object() && in.resolved.manifest.contains("process_helpers") &&
        !in.resolved.manifest.at("process_helpers").empty()) {
        const Json& ph = in.resolved.manifest.at("process_helpers");
        std::string arrangement;
        bool heads_then_one_tail = ph.size() >= 2;
        for (size_t i = 0; i < ph.size(); ++i) {
            const std::string where = ph.at(i).value("placement", "");
            arrangement += (i ? "," : "") + where;
            if (where != (i + 1 == ph.size() ? "tail" : "head")) heads_then_one_tail = false;
        }
        if (!heads_then_one_tail) {
            if (skipped_reason)
                *skipped_reason = "the live placement planner needs head helpers followed by exactly one tail helper; this layout's "
                                  "helpers are [" + arrangement + "] (no tail helper, or heads weighted by share, are not supported yet)";
            return false;
        }
    }
    return true;
}
} // namespace

std::optional<DerivedLivePolicy> derive_manifest_live_policy(
    const ManifestInputs& in, const bc::experts::Container& container, const Json& spec,
    std::string* skipped_reason) {
    // An unmeasured manifest is never what a start uses (the measured one follows), so it carries no reason.
    if (!in.measured) return std::nullopt;
    if (!planner_arrangement_supported(in, skipped_reason)) return std::nullopt;
    if (in.calibration_cache.empty()) {
        if (skipped_reason) *skipped_reason = "no calibration store was given (bc-serve --calibration)";
        return std::nullopt;
    }
    const Json policy_inputs = policy_calibration_inputs(spec);
    const std::string key = policy_calibration_key(policy_inputs);
    const auto record = read_live_policy_record(in.calibration_cache, key);
    if (!record) {
        // Silence here once left every start of a layout without a policy after a plan-key change orphaned its record.
        if (skipped_reason)
            *skipped_reason = "no calibration record for key=" + key + " in " + in.calibration_cache.string();
        return std::nullopt;
    }
    const auto skip = [&](const std::string& why) -> std::optional<DerivedLivePolicy> {
        if (skipped_reason) *skipped_reason = "record key=" + key + ": " + why;
        return std::nullopt;
    };
    if (!record->calibration_inputs || *record->calibration_inputs != policy_inputs)
        return skip("record has no matching model/card/service calibration provenance; not activating incompatible evidence");
    // Behind a TP engine (RAD49) every rank counts every routed row of a handed-off pass (the router is replicated), so rank
    // 0's histogram covers the layers either rank consumes, and the helpers store and move full-width experts: the policy
    // applies to any engine width. The histogram counts only passes the remote row gate admits (the publish op runs only
    // there, R3_MOE_REMOTE_MIN_ROWS / MAX_ROWS), as for a one-rank engine: run the policy on a workload whose steps the gate
    // admits.
    if (!live_policy_record_complete(*record)) {
        for (auto it = record->absent.begin(); it != record->absent.end(); ++it)
            return skip("incomplete " + it.key() + ": " + it.value().get<std::string>());
        return skip("incomplete calibration record");
    }
    // A structurally valid record may still be unusable on this layout or
    // statistically unresolvable. Those are policy failures, not serving failures.
    // Keep the strict file/schema reader OUTSIDE this catch.
    try {
        // The GPU backend allocates fixed-size slots. Select the largest *device
        // layout* expert among the container's supported expert descriptors, not
        // their compressed file bytes, and reject a layout with no usable experts.
        uint64_t stride = 0;
        for (uint32_t layer = 0; layer < container.layers; ++layer)
            for (uint32_t expert = 0; expert < container.experts; ++expert) {
                if (!container.weight(layer, expert, 0) || !container.weight(layer, expert, 1))
                    continue;
                const int64_t bytes = bc::experts::laid_out_bytes(container, true, layer, expert);
                if (bytes <= 0) throw std::runtime_error("live policy: cannot derive a device-layout expert stride");
                stride = std::max(stride, uint64_t(bytes));
            }
        if (!stride) throw std::runtime_error("live policy: no valid experts in the container");
        std::vector<uint64_t> budgets;
        for (const auto& h : in.resolved.manifest.at("process_helpers")) {
            const auto mib = h.at("budget_mib").get<uint64_t>();
            if (!mib || mib > std::numeric_limits<uint64_t>::max() / 1048576ull)
                throw std::runtime_error("live policy: invalid helper budget");
            budgets.push_back(mib * 1048576ull);
        }
        const CalibrationMatch match{"", in.resolved.values.at("layout").at("name").get<std::string>(),
                                      in.resolved.values.at("service").at("objective").get<std::string>()};
        return derive_live_policy(*record, Calibration::standard(in.calibration_cache), match, stride, budgets);
    } catch (const std::exception& e) {
        return skip(e.what());
    }
}

std::optional<Json> derive_manifest_live_calibration(const ManifestInputs& in, const Json& spec, std::string* why) {
    const auto no = [&](const std::string& reason) -> std::optional<Json> {
        if (why) *why = reason;
        return std::nullopt;
    };
    if (!in.measured || !in.auto_calibrate) return std::nullopt;
    if (!planner_arrangement_supported(in, why)) return std::nullopt;
    if (in.calibration_cache.empty()) return no("auto-calibration needs a calibration store (bc-serve --calibration)");
    // A store the session cannot write could never keep the record: do not measure for nothing.
    const auto dir = in.calibration_cache.has_parent_path() ? in.calibration_cache.parent_path() : std::filesystem::path(".");
    if (::access(dir.c_str(), W_OK) != 0)
        return no("auto-calibration needs a writable calibration store directory: " + dir.string());
    const Json& telemetry = spec.at("handoff").at("telemetry");
    if (!telemetry.contains("histogram_file") || telemetry.at("histogram_file").is_null() || telemetry.at("period_s").get<int64_t>() <= 0)
        return no("auto-calibration needs the live heat histogram on (handoff.histogram with a period)");
    try {
        const std::string key = policy_calibration_key(policy_calibration_inputs(spec));
        const auto record = read_live_policy_record(in.calibration_cache, key);
        if (record && live_policy_record_complete(*record)) return std::nullopt;   // derive_manifest_live_policy owns this case
        const CalibrationMatch match{"", in.resolved.values.at("layout").at("name").get<std::string>(),
                                     in.resolved.values.at("service").at("objective").get<std::string>()};
        const Calibration cal = Calibration::standard(in.calibration_cache);
        const int64_t windows = cal.integer("calibration.live_policy.transaction_effect_windows", match);
        const int64_t false_rate = cal.integer("calibration.live_policy.false_revert_rate_ppm", match);
        const int64_t missed = cal.integer("calibration.live_policy.calibration_missed_heat_writes", match);
        // The heat file is rewritten once per period: a probe may use an observation up to the declared number of missed
        // writes older than that.
        const int64_t period_ns = telemetry.at("period_s").get<int64_t>() * 1000000000LL;
        if (windows <= 0 || false_rate <= 0 || false_rate >= 1000000 || missed < 0 ||
            period_ns > std::numeric_limits<int64_t>::max() / (missed + 1))
            return no("declared calibration objectives are out of range");
        return Json{{"effect_windows", windows}, {"false_revert_rate_ppm", false_rate},
                    {"max_heat_age_ns", period_ns * (missed + 1)}};
    } catch (const std::exception& e) {
        return no(std::string("auto-calibration unavailable: ") + e.what());
    }
}
} // namespace bc::serving
