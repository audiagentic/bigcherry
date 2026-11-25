#include "engine_admission.h"

#include "serving_codec.h"
#include "serving_types.h"
#include "wire_reader.h"

namespace bc::serving {
namespace {
using Json = nlohmann::json;
using wire::Fault;
using wire::Wire;

bool env_name_bound(const std::string& name) {
    for (const v2::EnvBinding& b : v2::kEnvBindings)
        if (b.env == name) return true;
    return false;
}
}  // namespace

Json engine_admission_to_json(const EngineAdmission& a) {
    Json settings = {{"handoff", a.handoff}, {"skip_owned", a.skip_owned}, {"shadow", a.shadow}, {"combine", a.combine},
                     {"mask_only", a.mask_only}, {"mailbox_trace_every", a.mailbox_trace_every},
                     {"max_rows", static_cast<uint64_t>(a.max_rows)}, {"histogram_period_s", static_cast<uint64_t>(a.histogram_period_s)}, {"shm", a.shm},
                     {"histogram_file", a.histogram_file ? Json(*a.histogram_file) : Json(nullptr)},
                     {"moe_native", a.moe_native}, {"moe_fp16", a.moe_fp16}, {"moe_ahalf", a.moe_ahalf},
                     {"gemv", a.gemv}, {"dec11", a.dec11}, {"fuse_stage", a.fuse_stage},
                     {"attn_native", a.attn_native}, {"all_reduce_wire4", a.all_reduce_wire4}};
    Json env = Json::object();
    for (const auto& [name, value] : a.env) env[name] = value;
    return {{"kind", "bigcherry.engine-admission/v1"},
            {"measuring", a.measuring},
            {"bdfs", a.bdfs},
            {"rank_devices", a.rank_devices ? Json(*a.rank_devices) : Json(nullptr)},
            {"host", {{"pool_mib_per_rank", a.host_pool_mib_per_rank}, {"kv_mib", a.host_kv_mib},
                      {"reserve_mib", a.host_reserve_mib},
                      {"checkpoint_slots", a.checkpoint_slots ? Json(*a.checkpoint_slots) : Json(nullptr)}}},
            {"kv_backing_floor_ppm", a.kv_backing_floor_ppm},
            {"expected", a.expected ? *a.expected : Json(nullptr)},
            {"settings", settings},
            {"env", env}};
}

v2::Result<EngineAdmission> engine_admission_from_json(const Json& j) {
    try {
        Wire w(j, "engine_admission");
        if (w.text("kind") != "bigcherry.engine-admission/v1")
            throw Fault{v2::ConfigErrorCode::invalid_value, "engine_admission.kind", "not an engine admission document"};
        EngineAdmission a;
        a.measuring = w.boolean("measuring");
        const auto strings = [&](const Json& v, const std::string& path) {
            if (!v.is_array()) throw Fault{v2::ConfigErrorCode::invalid_type, path, "expected an array of strings"};
            std::vector<std::string> out;
            for (const Json& e : v) {
                if (!e.is_string()) throw Fault{v2::ConfigErrorCode::invalid_type, path, "expected an array of strings"};
                out.push_back(e.get<std::string>());
            }
            return out;
        };
        a.bdfs = strings(w.at("bdfs"), w.path("bdfs"));
        if (a.bdfs.empty()) throw Fault{v2::ConfigErrorCode::invalid_value, w.path("bdfs"), "no engine device"};
        if (const Json& r = w.at("rank_devices"); !r.is_null()) a.rank_devices = strings(r, w.path("rank_devices"));
        {
            Wire h(w.at("host"), "engine_admission.host");
            const auto size = [&](const char* key) {
                const Json& v = h.at(key);
                if (!v.is_number_integer() || v.get<int64_t>() < 0)
                    throw Fault{v2::ConfigErrorCode::invalid_type, h.path(key), "expected a non-negative integer"};
                return v.get<int64_t>();
            };
            a.host_pool_mib_per_rank = size("pool_mib_per_rank");
            a.host_kv_mib = size("kv_mib");
            a.host_reserve_mib = size("reserve_mib");
            if (const Json& c = h.at("checkpoint_slots"); !c.is_null()) {
                if (!c.is_number_integer() || c.get<int64_t>() < 0)
                    throw Fault{v2::ConfigErrorCode::invalid_type, h.path("checkpoint_slots"), "expected a non-negative integer"};
                a.checkpoint_slots = c.get<int64_t>();
            }
            h.done();
        }
        const Json& floor = w.at("kv_backing_floor_ppm");
        if (!floor.is_number_integer() || floor.get<int64_t>() < 0)
            throw Fault{v2::ConfigErrorCode::invalid_type, w.path("kv_backing_floor_ppm"), "expected a non-negative integer"};
        a.kv_backing_floor_ppm = floor.get<int64_t>();
        if (const Json& e = w.at("expected"); !e.is_null()) {
            v2::EnginePlan plan;   // the same decode the manifest and the plan cache use
            if (const auto bad = v2::decode(e, w.path("expected"), &plan))
                throw Fault{bad->code, bad->path, bad->message};
            a.expected = e;
        }
        if (a.measuring == a.expected.has_value())
            throw Fault{v2::ConfigErrorCode::invalid_state, w.path("expected"),
                        a.measuring ? "a measuring start has no plan to admit" : "a serving start needs the plan to admit"};
        {
            Wire s(w.at("settings"), "engine_admission.settings");
            a.handoff = s.boolean("handoff");
            a.skip_owned = s.boolean("skip_owned");
            a.shadow = s.boolean("shadow");
            a.combine = s.boolean("combine");
            a.mask_only = s.u32("mask_only");
            a.mailbox_trace_every = s.u32("mailbox_trace_every");
            a.max_rows = static_cast<int64_t>(s.u64("max_rows", INT32_MAX));
            a.histogram_period_s = static_cast<int64_t>(s.u64("histogram_period_s", INT32_MAX));
            a.shm = s.text("shm");
            if (const Json& h = s.at("histogram_file"); !h.is_null()) {
                if (!h.is_string()) throw Fault{v2::ConfigErrorCode::invalid_type, s.path("histogram_file"), "expected a string"};
                a.histogram_file = h.get<std::string>();
            }
            a.moe_native = s.boolean("moe_native");
            a.moe_fp16 = s.boolean("moe_fp16");
            a.moe_ahalf = s.boolean("moe_ahalf");
            a.gemv = s.boolean("gemv");
            a.dec11 = s.boolean("dec11");
            a.fuse_stage = s.boolean("fuse_stage");
            a.attn_native = s.boolean("attn_native");
            a.all_reduce_wire4 = s.boolean("all_reduce_wire4");
            s.done();
        }
        const Json& env = w.at("env");
        if (!env.is_object()) throw Fault{v2::ConfigErrorCode::invalid_type, w.path("env"), "expected an object"};
        for (auto it = env.begin(); it != env.end(); ++it) {
            if (!it.value().is_string() || !env_name_bound(it.key()))
                throw Fault{v2::ConfigErrorCode::unknown_field, w.path("env") + "." + it.key(),
                            "not an engine switch the contract binds, or not a string"};
            a.env.emplace(it.key(), it.value().get<std::string>());
        }
        w.done();
        return a;
    } catch (const Fault& f) {
        return v2::make_error(f.code, f.path, f.message);
    }
}

v2::Result<EngineAdmission> engine_admission_from_manifest(const v2::Manifest& m, bool measuring) {
    const v2::Spec& spec = m.spec;
    EngineAdmission a;
    a.measuring = measuring;
    a.bdfs = spec.processes.engine.device.gpus;
    a.rank_devices = spec.processes.engine.device.rank_devices;
    a.host_pool_mib_per_rank = spec.resources.host.pool_mib;
    a.host_kv_mib = spec.resources.host.kv_mib;
    a.host_reserve_mib = spec.resources.host.reserve_mib;
    a.checkpoint_slots = spec.resources.host.checkpoint_slots;
    if (measuring) {
        for (const v2::CalibrationEntry& entry : spec.calibration)
            if (entry.key == v2::CalibrationKey::kKvBackingFloorFractionPpm)
                a.kv_backing_floor_ppm = std::get<int64_t>(entry.value);
    } else {
        const auto view = v2::engine_view(m);
        if (!view) return view.error();
        a.kv_backing_floor_ppm = view.value().kv_backing_floor_ppm;
        a.expected = v2::encode(view.value().memory.expected);
    }
    if (spec.handoff) {
        const auto& h = *spec.handoff;
        a.handoff = true;
        a.max_rows = h.transport.max_rows;
        a.skip_owned = h.transport.skip_owned;
        a.shm = h.transport.shm;
        a.histogram_file = h.telemetry.histogram_file;
        a.histogram_period_s = h.telemetry.period_s;
        a.combine = spec.processes.helper_common->combine.enabled;
    }
    if (spec.processes.engine.kernels) {
        const auto& k = *spec.processes.engine.kernels;
        a.moe_native = k.moe_native.value_or(true);
        a.moe_fp16 = k.moe_fp16.value_or(true);
        a.moe_ahalf = k.moe_ahalf.value_or(true);
        a.gemv = k.gemv.value_or(false);
        a.dec11 = k.dec11.value_or(false);
        a.fuse_stage = k.fuse_stage.value_or(true);
        a.attn_native = k.attn_native.value_or(true);
        a.all_reduce_wire4 = k.all_reduce_wire4.value_or(false);
    }
    if (spec.experiment && spec.experiment->engine) {
        a.shadow = spec.experiment->engine->shadow.value_or(false);
        a.mask_only = static_cast<uint32_t>(spec.experiment->engine->mask_only.value_or(0));
    }
    if (spec.experiment && spec.experiment->diagnostics && spec.experiment->diagnostics->lanes &&
        spec.experiment->diagnostics->lanes->trace_level)
        a.mailbox_trace_every = static_cast<uint32_t>(*spec.experiment->diagnostics->lanes->trace_level);
    // The engine switches that are still environment variables for their consumers (RSC11 types them): same bridge as
    // the manifest entry, minus the ones that are typed settings above.
    const Json doc = v2::encode(m);
    for (const v2::EnvBinding& b : v2::kEnvBindings) {
        if (b.env == "R3_MOE_NATIVE" || b.env == "R3_MOE_FP16" || b.env == "R3_MOE_AHALF" || b.env == "R3_GEMV" ||
            b.env == "R3_DEC11" || b.env == "R3_MOE_WORKER_FUSESTAGE" || b.env == "R3_MOE_WORKER_MASK_ONLY" ||
            b.env == "R3_ATTN_NATIVE" || b.env == "R3_AR_WIRE4")
            continue;
        const Json::json_pointer ptr{std::string(b.path)};
        if (!doc.contains(ptr)) continue;
        const Json& v = doc.at(ptr);
        if (b.kind == "boolean" && v.is_boolean())
            a.env.emplace(std::string(b.env), v.get<bool>() ? "1" : "0");
        else if ((b.kind == "integer" || b.kind == "int_enum") && v.is_number_integer())
            a.env.emplace(std::string(b.env), std::to_string(v.get<long long>()));
    }
    return a;
}

}  // namespace bc::serving
