#include "admission.h"

#include <algorithm>
#include <map>
#include <string>

namespace bc::serving::v1 {
namespace {

constexpr int64_t kMaxBytes = (int64_t(1) << 53) - 1;
const std::string kAt = "spec.plan.engine_memory.expected";
const char* const kMeasure = "measure the plan again (bc-serve --plan-memory) and resolve a new manifest";
const char* const kFree = "free the engine card, or choose a smaller service intent, and resolve a new manifest";

ConfigError differ(const std::string& at, const std::string& what, int64_t live, int64_t planned) {
    return make_error(ConfigErrorCode::plan_mismatch, at,
                      what + ": live " + std::to_string(live) + " bytes differ from the planned " +
                          std::to_string(planned) + " bytes",
                      kMeasure);
}

ConfigError short_of(const std::string& at, const std::string& what, int64_t live, int64_t required,
                     const std::string& detail) {
    return make_error(ConfigErrorCode::admission_refused, at,
                      what + ": live " + std::to_string(live) + " bytes fall short of the required " +
                          std::to_string(required) + " bytes (" + detail + ")",
                      kFree);
}

}  // namespace

int64_t kv_required(int64_t kv_ceiling, int64_t kv_one_session, int64_t ppm) {
    const int64_t floor = kv_ceiling / 1000000 * ppm + (kv_ceiling % 1000000 * ppm + 999999) / 1000000;
    return std::max(kv_one_session, floor);
}

Status check_engine_plan(const EnginePlan& expected, const EnginePlan& live, int64_t ppm) {
    if (ppm < 0 || ppm >= 1000000)
        return make_error(ConfigErrorCode::invalid_value, "spec.calibration",
                          "plan check: kv_backing_floor_ppm must be in [0, 1000000), is " + std::to_string(ppm),
                          "select a kv_backing_floor.fraction_ppm calibration in range");
    const PlanVram& ev = expected.vram;
    const PlanVram& lv = live.vram;
    for (const PlanVram* v : {&ev, &lv})
        for (int64_t t : {v->claimable, v->statics, v->experts, v->stage, v->kv_resolved, v->kv_ceiling, v->expert_floor,
                          v->expert_ceiling})
            if (t < 0 || t > kMaxBytes)
                return make_error(ConfigErrorCode::invalid_value, kAt + ".vram",
                                  "plan check: the VRAM terms must be byte counts in [0, 2^53 - 1]", kMeasure);
    if (expected.kv_one_session.bytes < 0 || expected.kv_one_session.bytes > kMaxBytes)
        return make_error(ConfigErrorCode::invalid_value, kAt + ".kv_one_session.bytes",
                          "plan check: the VRAM terms must be byte counts in [0, 2^53 - 1]", kMeasure);

    // Stable facts: what the weights, the model's geometry and the measured placement fix (spec §5.2).
    std::map<std::string, const PlanHostConsumer*> planned;
    for (const PlanHostConsumer& c : expected.host) planned[c.name] = &c;
    for (size_t i = 0; i < live.host.size(); ++i) {
        const PlanHostConsumer& c = live.host[i];
        const std::string at = kAt + ".host[" + std::to_string(i) + "]";
        const auto it = planned.find(c.name);
        if (it == planned.end())
            return make_error(ConfigErrorCode::plan_mismatch, at,
                              "host consumer " + c.name + " is not in the planned inventory",
                              c.remedy.empty() ? kMeasure : c.remedy);
        if (c.bytes != it->second->bytes) return differ(at, "host consumer " + c.name, c.bytes, it->second->bytes);
        if (c.residence != it->second->residence)
            return make_error(ConfigErrorCode::plan_mismatch, at,
                              "host consumer " + c.name + ": live residence " +
                                  std::string(kResidenceNames[static_cast<size_t>(c.residence)]) + " differs from the planned " +
                                  std::string(kResidenceNames[static_cast<size_t>(it->second->residence)]),
                              kMeasure);
    }
    if (live.host.size() != expected.host.size()) {
        for (const PlanHostConsumer& c : expected.host)
            if (std::none_of(live.host.begin(), live.host.end(), [&](const PlanHostConsumer& l) { return l.name == c.name; }))
                return make_error(ConfigErrorCode::plan_mismatch, kAt + ".host",
                                  "the planned host consumer " + c.name + " is missing from the live plan", kMeasure);
        return make_error(ConfigErrorCode::plan_mismatch, kAt + ".host", "a host consumer is repeated in the live plan",
                          kMeasure);
    }
    const PlanCheckpoints& ec = expected.checkpoints;
    const PlanCheckpoints& lc = live.checkpoints;
    if (lc.slot_bytes != ec.slot_bytes) return differ(kAt + ".checkpoints.slot_bytes", "checkpoint slot", lc.slot_bytes, ec.slot_bytes);
    if (lc.floor_slots != ec.floor_slots)
        return make_error(ConfigErrorCode::plan_mismatch, kAt + ".checkpoints.floor_slots",
                          "checkpoint floor: live " + std::to_string(lc.floor_slots) + " slots differ from the planned " +
                              std::to_string(ec.floor_slots),
                          kMeasure);
    if (lc.want_slots != ec.want_slots)
        return make_error(ConfigErrorCode::plan_mismatch, kAt + ".checkpoints.want_slots",
                          "checkpoint want: live " + std::to_string(lc.want_slots) + " slots differ from the planned " +
                              std::to_string(ec.want_slots),
                          kMeasure);
    if (live.kv_one_session.bytes != expected.kv_one_session.bytes)
        return differ(kAt + ".kv_one_session.bytes", "one full-length session's KV", live.kv_one_session.bytes,
                      expected.kv_one_session.bytes);
    if (live.host_pool_mib != expected.host_pool_mib)
        return make_error(ConfigErrorCode::plan_mismatch, kAt + ".host_pool_mib",
                          "host pool: live " + std::to_string(live.host_pool_mib) + " MiB differs from the planned " +
                              std::to_string(expected.host_pool_mib) + " MiB",
                          kMeasure);
    const std::pair<const char*, int64_t PlanVram::*> stable[] = {{"statics", &PlanVram::statics},
                                                                  {"expert_floor", &PlanVram::expert_floor},
                                                                  {"expert_ceiling", &PlanVram::expert_ceiling},
                                                                  {"kv_ceiling", &PlanVram::kv_ceiling}};
    for (const auto& [name, field] : stable)
        if (lv.*field != ev.*field)
            return differ(kAt + ".vram." + name, std::string("VRAM ") + name, lv.*field, ev.*field);
    if (lv.cards.size() != ev.cards.size())
        return make_error(ConfigErrorCode::plan_mismatch, kAt + ".vram.cards",
                          "plan check: the live plan has " + std::to_string(lv.cards.size()) + " card(s), the manifest's " +
                              std::to_string(ev.cards.size()),
                          kMeasure);
    for (size_t i = 0; i < lv.cards.size(); ++i) {
        const std::string at = kAt + ".vram.cards[" + std::to_string(i) + "]";
        const std::string card = "card " + std::to_string(i) + " ";
        if (lv.cards[i].capacity < ev.cards[i].capacity)
            return make_error(ConfigErrorCode::plan_mismatch, at + ".capacity",
                              card + "capacity: live " + std::to_string(lv.cards[i].capacity) +
                                  " bytes fall short of the planned " + std::to_string(ev.cards[i].capacity) + " bytes",
                              kMeasure);
        if (lv.cards[i].engine_expert_plane != ev.cards[i].engine_expert_plane)
            return differ(at + ".engine_expert_plane", card + "engine_expert_plane", lv.cards[i].engine_expert_plane,
                          ev.cards[i].engine_expert_plane);
        if (lv.cards[i].expert_unit_max != ev.cards[i].expert_unit_max)
            return differ(at + ".expert_unit_max", card + "expert_unit_max", lv.cards[i].expert_unit_max,
                          ev.cards[i].expert_unit_max);
    }

    // Admission on the live split; the measured kv share is never an input (amendment 5).
    const int64_t need = kv_required(ev.kv_ceiling, expected.kv_one_session.bytes, ppm);
    if (lv.experts < ev.expert_floor)
        return short_of(kAt + ".vram.experts", "admission: expert residency", lv.experts, ev.expert_floor,
                        "the experts the pinned host pool cannot hold");
    const int64_t total = ev.statics + lv.experts + lv.stage + need;
    if (lv.claimable < total)
        return short_of(kAt + ".vram.claimable", "admission: claimable VRAM", lv.claimable, total,
                        "statics " + std::to_string(ev.statics) + " + experts " + std::to_string(lv.experts) +
                            " + stage " + std::to_string(lv.stage) + " + kv_required " + std::to_string(need));
    if (lv.kv_resolved < need)
        return short_of(kAt + ".vram.kv_resolved", "admission: resolved KV", lv.kv_resolved, need,
                        "kv_required " + std::to_string(need) +
                            ", the common KV budget after the staging split and MiB rounding");
    return std::nullopt;
}

}  // namespace bc::serving::v1
