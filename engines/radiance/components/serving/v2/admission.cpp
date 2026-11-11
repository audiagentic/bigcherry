#include "admission.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace bc::serving::v2 {
namespace {

constexpr int64_t kMaxBytes = (int64_t(1) << 53) - 1;
const std::string kAt = "spec.plan.engine_memory.expected";
const char* const kMeasure = "measure the plan again (the launcher's measuring start: the engine's --plan-memory with its helpers up) and write a new manifest from it (bc-serve --write-manifest --measured-plan)";
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

/* Build the S11 room payload from the same measured terms used by admission.
 * An offline check without an authoritative manifest BDF retains the typed
 * ConfigError but cannot advertise a retryable GPU refusal. */
ConfigError gpu_room(const std::string& at, const std::string& what,
                     int64_t available, int64_t statics, int64_t experts, int64_t stage,
                     int64_t kv, std::string_view bdf) {
    const int64_t required = statics + experts + stage + kv;
    if (bdf.empty())
        return short_of(at, what, available, required,
                        "statics " + std::to_string(statics) + " + experts " +
                            std::to_string(experts) + " + stage " + std::to_string(stage) +
                            " + kv_required " + std::to_string(kv));
    ConfigError e = make_error(ConfigErrorCode::admission_refused, at, "", kFree);
    e.room_refusal = RoomRefusal{"gpu_room", "vram", std::string(bdf), available,
                                required, 0,
                                {{"statics", statics}, {"experts", experts},
                                 {"stage", stage}, {"kv_required", kv}},
                                {"gpu_elastic_kv"}};
    return e;
}

}  // namespace

Status check_host_room(std::string_view resource, int64_t available_bytes, int64_t headroom_bytes,
                       const std::vector<std::pair<std::string, int64_t>>& components) {
    const bool pin = resource == "host_pool";
    if (!pin && resource != "host_ram" && resource != "host_kv")
        return make_error(ConfigErrorCode::invalid_value, "host_room.resource",
                          "unknown S5 host resource", "measure a supported host resource");
    if (available_bytes < 0 || headroom_bytes < 0 || available_bytes > kMaxBytes ||
        headroom_bytes > kMaxBytes || components.empty())
        return make_error(ConfigErrorCode::invalid_value, "host_room",
                          "host memory bound/components missing or invalid", kMeasure);
    int64_t remaining = 0;
    for (const auto& [key, bytes] : components) {
        if (key.empty() || bytes < 0 || bytes > kMaxBytes - remaining)
            return make_error(ConfigErrorCode::invalid_value, "host_room.components",
                              "invalid host memory component", kMeasure);
        remaining += bytes;
    }
    if (headroom_bytes > kMaxBytes - remaining)
        return make_error(ConfigErrorCode::invalid_value, "host_room.headroom",
                          "host memory requirement overflows the measured byte range", kMeasure);
    if (remaining + headroom_bytes <= available_bytes) return std::nullopt;
    ConfigError e = make_error(ConfigErrorCode::admission_refused, "host_room",
                               "", "remeasure the host plan and release host KV; otherwise free the named shortfall");
    e.room_refusal = RoomRefusal{pin ? "pin_room" : "host_room",
                                std::string(resource), std::nullopt,
                                available_bytes, remaining, headroom_bytes,
                                components,
                                pin ? std::vector<std::string>{"host_kv"}
                                    : std::vector<std::string>{"host_ledger_margin", "host_kv"}};
    return e;
}

int64_t kv_required(int64_t kv_ceiling, int64_t kv_one_session, int64_t ppm) {
    const int64_t floor = kv_ceiling / 1000000 * ppm + (kv_ceiling % 1000000 * ppm + 999999) / 1000000;
    return std::max(kv_one_session, floor);
}

Status check_kv_backing_floor_ppm(int64_t ppm) {
    if (ppm < 0 || ppm >= 1000000)
        return make_error(ConfigErrorCode::invalid_value, "spec.calibration",
                          "plan check: kv_backing_floor_ppm must be in [0, 1000000), is " + std::to_string(ppm),
                          "select a kv_backing_floor.fraction_ppm calibration in range");
    return std::nullopt;
}

AdmissionMargins admission_margins(const EnginePlan& expected, const EnginePlan& live, int64_t ppm) {
    const int64_t need = kv_required(expected.vram.kv_ceiling, expected.kv_one_session.bytes, ppm);
    return {need, expected.vram.statics + live.vram.experts + live.vram.stage + need, live.vram.claimable,
            live.vram.kv_resolved};
}

Status check_engine_plan(const EnginePlan& expected, const EnginePlan& live, int64_t ppm,
                         std::string_view device_bdf) {
    if (auto e = check_kv_backing_floor_ppm(ppm)) return e;
    const PlanVram& ev = expected.vram;
    const PlanVram& lv = live.vram;
    // Input: every count this check reads is a byte or slot count, every enum in range, the checkpoint floor within
    // the want; before any comparison, so a malformed plan is never reported as drift.
    for (const auto& [plan, where] : {std::pair<const EnginePlan*, const char*>{&expected, "expected"}, {&live, "live"}}) {
        const PlanVram& v = plan->vram;
        std::vector<int64_t> counts = {v.claimable, v.statics, v.elastic, v.experts, v.kv, v.stage, v.kv_resolved,
                                       v.expert_floor, v.kv_ceiling, v.expert_ceiling, plan->kv_one_session.bytes,
                                       plan->checkpoints.slot_bytes, plan->checkpoints.floor_slots,
                                       plan->checkpoints.want_slots, plan->host_pool_mib};
        for (const PlanHostConsumer& c : plan->host) counts.push_back(c.bytes);
        for (const PlanCard& c : v.cards)
            for (int64_t t : {c.capacity, c.free, c.engine_expert_plane, c.expert_unit_max}) counts.push_back(t);
        for (int64_t t : counts)
            if (t < 0 || t > kMaxBytes)
                return make_error(ConfigErrorCode::invalid_value, kAt,
                                  std::string("plan check: the ") + where +
                                      " plan's VRAM terms must be byte counts in [0, 2^53 - 1]",
                                  kMeasure);
        for (const PlanHostConsumer& c : plan->host)
            if (static_cast<size_t>(c.residence) >= kResidenceNames.size())
                return make_error(ConfigErrorCode::invalid_value, kAt + ".host",
                                  std::string("plan check: the ") + where + " plan has an unknown residence", kMeasure);
        if (plan->checkpoints.floor_slots > plan->checkpoints.want_slots)
            return make_error(ConfigErrorCode::invalid_value, kAt + ".checkpoints",
                              std::string("plan check: the ") + where + " plan's floor_slots exceed its want_slots",
                              kMeasure);
    }

    // Stable facts: what the weights, the model's geometry and the measured placement fix (spec §5.2). The host
    // inventory matches exactly: the same names, each once, with the same bytes and residence.
    std::map<std::string, const PlanHostConsumer*> planned, seen;
    for (const PlanHostConsumer& c : expected.host)
        if (!planned.emplace(c.name, &c).second)
            return make_error(ConfigErrorCode::plan_mismatch, kAt + ".host",
                              "host consumer " + c.name + " is repeated in the planned inventory", kMeasure);
    for (size_t i = 0; i < live.host.size(); ++i) {
        const PlanHostConsumer& c = live.host[i];
        const std::string at = kAt + ".host[" + std::to_string(i) + "]";
        if (!seen.emplace(c.name, &c).second)
            return make_error(ConfigErrorCode::plan_mismatch, at,
                              "host consumer " + c.name + " is repeated in the live inventory", kMeasure);
        const auto it = planned.find(c.name);
        if (it == planned.end())
            return make_error(ConfigErrorCode::plan_mismatch, at,
                              "host consumer " + c.name + " is not in the planned inventory",
                              c.remedy.empty() ? kMeasure : c.remedy);
        if (c.bytes != it->second->bytes) return differ(at, "host consumer " + c.name, c.bytes, it->second->bytes);
        if (c.residence != it->second->residence)
            return make_error(ConfigErrorCode::plan_mismatch, at,
                              "host consumer " + c.name + ": live residence " +
                                  std::string(kResidenceNames[static_cast<size_t>(c.residence)]) +
                                  " differs from the planned " +
                                  std::string(kResidenceNames[static_cast<size_t>(it->second->residence)]),
                              kMeasure);
    }
    for (const PlanHostConsumer& c : expected.host)
        if (!seen.count(c.name))
            return make_error(ConfigErrorCode::plan_mismatch, kAt + ".host",
                              "the planned host consumer " + c.name + " is missing from the live plan", kMeasure);
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
        return gpu_room(kAt + ".vram.claimable", "admission: claimable VRAM", lv.claimable,
                        ev.statics, lv.experts, lv.stage, need, device_bdf);
    if (lv.kv_resolved < need) {
        // The KV share after staging is the tighter binding constraint even if
        // raw claimable VRAM fits. Express the effective allocation ceiling in
        // the same units as total required, so S11's GPU refit is meaningful.
        if (!device_bdf.empty())
            return gpu_room(kAt + ".vram.kv_resolved", "admission: resolved KV",
                            ev.statics + lv.experts + lv.stage + lv.kv_resolved,
                            ev.statics, lv.experts, lv.stage, need, device_bdf);
        return short_of(kAt + ".vram.kv_resolved", "admission: resolved KV", lv.kv_resolved, need,
                        "kv_required " + std::to_string(need) +
                            ", the common KV budget after the staging split and MiB rounding");
    }
    return std::nullopt;
}

}  // namespace bc::serving::v2
