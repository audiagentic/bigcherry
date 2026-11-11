#include "manifest_writer.h"

#include "calibration.h"
#include "experts.h"
#include "jcs.h"
#include "ledger.h"
#include "live_policy.h"
#include "owner_map.h"
#include "serving_codec.h"
#include "sha256.h"
#include "third_party/toml.hpp"
#include "v2/manifest.h"

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace bc::serving {
namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;

// serving/helper_profile.toml, embedded at build time (CMakeLists.txt writes helper_profile_seed.inc).
constexpr const char kHelperProfile[] =
#include "helper_profile_seed.inc"
    ;

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error(what); }

std::string slurp(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) fail("cannot read " + p.string());
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

Json record(const char* origin, const char* source, const char* cls, const char* confidence,
            Json inputs = Json::array()) {
    return {{"origin", origin}, {"source", source}, {"class", cls}, {"lifecycle", "startup_only"},
            {"confidence", confidence}, {"inputs", std::move(inputs)}};
}

std::vector<std::string> split(const std::string& text, char sep) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : text) {
        if (c == sep) { out.push_back(cur); cur.clear(); }
        else cur += c;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

/* spec.processes.helper_common from helper_profile.toml: one row a leaf, "group/field". */
Json helper_common(const Json& values) {
    Json out = Json::object();
    toml::table doc;
    try { doc = toml::parse(std::string_view(kHelperProfile), std::string_view("helper_profile.toml")); }
    catch (const toml::parse_error& e) { fail(std::string("helper_profile.toml: ") + e.what()); }
    const auto* rows = doc["row"].as_array();
    if (!rows) fail("helper_profile.toml: no rows");
    for (const toml::node& n : *rows) {
        const auto* row = n.as_table();
        const auto* leaf = row ? (*row)["leaf"].as_string() : nullptr;
        const toml::node* value = row ? row->get("value") : nullptr;
        if (!leaf || !value) fail("helper_profile.toml: a row needs leaf and value");
        const std::vector<std::string> path = split(leaf->get(), '/');
        if (path.size() != 2) fail("helper_profile.toml: leaf " + leaf->get() + " is not group/field");
        Json v;
        if (auto b = value->as_boolean()) v = b->get();
        else if (auto i = value->as_integer()) v = i->get();
        else if (auto s = value->as_string()) v = s->get();
        else if (auto a = value->as_array(); a && a->empty()) v = Json::array();
        else fail("helper_profile.toml: leaf " + leaf->get() + " has a value of no supported kind");
        // A row naming a resolver key takes the resolved value (layout, site file or --set), not the table's default.
        if (const auto* from = (*row)["from"].as_string()) {
            const std::vector<std::string> key = split(from->get(), '.');
            if (key.size() != 2 || !values.contains(key[0]) || !values.at(key[0]).contains(key[1]))
                fail("helper_profile.toml: leaf " + leaf->get() + " names resolver key " + from->get() + " which is not resolved");
            v = values.at(key[0]).at(key[1]);
        }
        out[path[0]][path[1]] = std::move(v);
    }
    return out;
}

Json facts_of(const Json& f) {
    const Json& h = f.at("host");
    Json cgroups = Json::array();
    for (const Json& c : h.at("memory_cgroups"))
        cgroups.push_back({{"path", c.at("dir")}, {"room_mib", c.at("room_mib")}});
    Json gpus = Json::array();
    for (const Json& g : f.at("gpus"))
        gpus.push_back({{"bdf", g.at("bdf")}, {"arch", g.at("arch")}, {"hip_ordinal", g.at("hip_ordinal")},
                        {"vram", {{"total_mib", g.at("vram_total_mib")}, {"available_mib", g.at("vram_available_mib")}}}});
    const Json& pin = h.at("amdgpu_pin");
    return {{"source", f.at("source")},
            {"host",
             {{"memory",
               {{"total_mib", h.at("mem_total_mib")}, {"available_mib", h.at("mem_available_mib")},
                {"swap_total_mib", h.at("swap_total_mib")}, {"shm_available_mib", h.at("shm_available_mib")},
                {"cgroup_limits", cgroups},
                {"pin_cap_mib", pin.at("lifted").get<bool>() || !pin.at("loaded").get<bool>() ? Json(nullptr) : pin.at("cap_mib")},
                {"memlock_limit_mib", h.at("memlock").at("limit_mib")}}},
              {"cpu", {{"count", h.at("cpu_count")}}}}},
            {"gpus", gpus}};
}

Residence residence_of(v2::Residence r) {
    switch (r) {
        case v2::Residence::kPinned: return Residence::pinned;
        case v2::Residence::kMlocked: return Residence::mlocked;
        case v2::Residence::kPageable: return Residence::pageable;
    }
    return Residence::pageable;
}

Json calibration_entries(const Resolved& r, const fs::path& cache) {
    const Calibration cal = Calibration::standard(cache);
    const CalibrationMatch at{"", r.values.at("layout").at("name").get<std::string>(),
                              r.values.at("service").at("objective").get<std::string>()};
    Json out = Json::array();
    for (std::string_view key : v2::kCalibrationKeyNames) {
        const auto c = cal.find("calibration." + std::string(key), at);
        if (!c) continue;   // a parameter the objective does not use (e.g. no KV backing floor for throughput)
        out.push_back({{"key", key}, {"value", c->value}, {"source", c->source}, {"seeded", c->seeded}, {"evidence", c->evidence}});
    }
    return out;
}

/* spec.resources.host: the values the launcher runs with, and the checkpoint slots the RSC03 host ledger plans for
 * them (measured consumers and checkpoint store from the plan, the calibrated share and host KV tier target). */
Json resources_of(const ManifestInputs& in, const Json& calibration) {
    const Json& mem = in.resolved.values.at("memory");
    Json host = {{"pool_mib", mem.at("host_pool_mib")}, {"kv_mib", mem.at("host_kv_mib")}, {"reserve_mib", mem.at("host_reserve_mib")}};
    if (!in.measured) return host;   // no plan, no consumers to ledger: the checkpoint slots follow from the measured plan
    auto cal = [&](const char* key) -> const Json* {
        for (const Json& e : calibration)
            if (e.at("key") == key) return &e.at("value");
        return nullptr;
    };
    LedgerInputs li;
    li.host_facts = in.resolved.facts.at("host");
    li.reserve_mib = mem.at("host_reserve_mib");
    for (const v2::PlanHostConsumer& c : in.measured->plan.plan.host)
        li.consumers.push_back({c.name, static_cast<uint64_t>(c.bytes), residence_of(c.residence), c.from, c.remedy});
    // What the engine allocates and its neighbours hold besides the plan's own consumers (RSC08 S3: before this the ledger
    // planned slots the host could not hold, and the engine stopped at start): the host pool the launcher runs with (pinned;
    // the plan echoes it and the check demands it equal), and the host memory the helpers and the engine itself held in the
    // measuring start that produced the plan (pageable, measured: stored with the cache entry; MemAvailable above is read now).
    constexpr uint64_t kMiB = 1048576;
    const bool has_helpers = !in.resolved.manifest.at("process_helpers").empty();
    if ((has_helpers && in.measured->plan.resident.helpers_mib == 0) || in.measured->plan.resident.engine_mib == 0)
        fail("the measured plan has no resident host memory of the helpers and the engine: measure it again");
    // radiance pins --host-pool-mib once per rank (engine_bringup.cpp: "is pinned once per rank"), so a TP engine holds tp pools.
    const uint64_t ranks = in.resolved.values.at("engine").at("tp").get<uint64_t>();
    li.consumers.push_back({"host_pool", ranks * mem.at("host_pool_mib").get<uint64_t>() * kMiB, Residence::pinned,
                            "resources.host.pool_mib x engine.tp (the launcher's pool; the engine pins it once per rank at start)",
                            "lower memory.host_pool_mib"});
    li.consumers.push_back({"helpers", in.measured->plan.resident.helpers_mib * kMiB, Residence::pageable,
                            "measured in the measuring start: the helpers' resident anonymous and shared memory", ""});
    li.consumers.push_back({"engine_start", in.measured->plan.resident.engine_mib * kMiB, Residence::pageable,
                            "measured in the measuring start: what the engine's start-up took from MemAvailable before plan() ran", ""});
    // Memory left unplanned for what the host loses between this write and the engine's own check (calibrated; counted as a
    // pageable consumer so it comes out of every limit the ledger checks).
    if (const Json* margin = cal("host_ledger_margin.mib"))
        li.consumers.push_back({"drift_margin", margin->get<uint64_t>() * kMiB, Residence::pageable,
                                "calibration.host_ledger_margin.mib: what the host loses between the manifest write and the engine's check", ""});
    li.checkpoints = {static_cast<uint64_t>(in.measured->plan.plan.checkpoints.slot_bytes), static_cast<uint64_t>(in.measured->plan.plan.checkpoints.floor_slots),
                      static_cast<uint64_t>(in.measured->plan.plan.checkpoints.want_slots), in.measured->plan.plan.checkpoints.from, "want", "plan"};
    if (const Json* share = cal("checkpoint_share")) {
        li.checkpoints.share = share->get<std::string>();
        li.checkpoints.share_from = "calibration.checkpoint_share";
    }
    li.retention_keep = in.resolved.values.at("service").at("retention") == "keep";
    li.min_tier_bytes = static_cast<uint64_t>(in.measured->plan.plan.kv_one_session.bytes);
    // One value for the host KV tier: the manifest's resources.host.kv_mib is what the launcher runs with, so the ledger
    // plans for exactly that. A calibration target that disagrees would make the ledger plan for a tier the run does not
    // get; refuse it rather than pick one.
    const uint64_t kv_mib = mem.at("host_kv_mib").get<uint64_t>();
    if (const Json* target = cal("host_kv_tier.target_mib"))
        if (target->get<uint64_t>() != kv_mib)
            fail("resources.host.kv_mib (" + std::to_string(kv_mib) + ") differs from calibration host_kv_tier.target_mib (" +
                 std::to_string(target->get<uint64_t>()) + "): the memory plan would not match the run");
    li.tier_target_bytes = kv_mib << 20;
    const Ledger ledger = host_ledger(li);
    host["checkpoint_slots"] = ledger.checkpoint_slots;
    return host;
}

Json handoff_of(const Json& v, const Json& live_policy) {
    const Json& h = v.at("handoff");
    Json out = {{"transport",
             {{"shm", h.at("shm")}, {"max_rows", h.at("max_rows")}, {"small_rows", h.at("small_rows")},
              {"gemv_large_rows", h.at("gemv_large_rows")}, {"skip_owned", h.at("skip_owned")}}},
            {"placement", {{"strategy", h.at("placement")}, {"heat_file", h.at("heat")}}},
            {"telemetry",
             {{"histogram_file", h.at("histogram").get<std::string>().empty() ? Json(nullptr) : h.at("histogram")},
              {"period_s", h.at("histogram_period_s")}}},
            {"kernels", {{"moe_down_tile", h.at("moe_down_tile").get<bool>() ? 1 : 0}}}};
    if (!live_policy.is_null()) out["placement"]["live_policy"] = live_policy;
    return out;
}

// The draft plan's devices: the engine TP cards and the selected helper cards, each with no settings. S10 resolves them.
Json draft_power_devices(const Resolved& r) {
    Json out = Json::array();
    for (const Json& bdf : r.values.at("layout").at("engine_cards"))
        out.push_back({{"role", "engine"}, {"bdf", bdf}, {"settings", Json::object()}});
    const Json& ph = r.manifest.at("process_helpers");
    for (size_t i = 0; i < ph.size(); ++i)
        out.push_back({{"role", "helper" + std::to_string(i)}, {"bdf", ph[i].at("bdf")}, {"settings", Json::object()}});
    return out;
}

Json engine_of(const Json& v) {
    const Json& e = v.at("engine");
    Json device = {{"gpus", v.at("layout").at("engine_cards")}};
    if (e.contains("gpu_headroom_mib")) device["headroom_mib"] = e.at("gpu_headroom_mib");
    Json execution = {{"placement", e.at("placement")}, {"ngram_placement", e.at("ngram_placement")},
                      {"kv_cache_dtype", e.at("kv_cache_dtype")}, {"num_speculative_tokens", e.at("num_speculative_tokens")},
                      {"rope_scaling", e.at("rope_scaling")}, {"prefix_resume", e.at("prefix_resume")}};
    if (e.contains("tp_wire")) execution["tp_wire"] = e.at("tp_wire");
    if (e.contains("no_prefix_cache")) execution["no_prefix_cache"] = e.at("no_prefix_cache");
    Json out_engine = {{"device", device},
            {"runtime", {{"radiance_home", e.at("radiance_home")}, {"kernels", split(e.at("kernels"), ',')}}},
            {"server", {{"port", e.at("port")}}},
            {"scheduler", {{"tp", e.at("tp")}, {"max_num_batched_tokens", e.at("max_num_batched_tokens")},
                           {"checkpoint_interval", e.at("checkpoint_interval")}}},
            {"execution", execution}};
    // Amendment 13: the engine options with a field of their own, each section present only with the members that are set.
    for (const auto& o : engine_option_fields()) {
        if (!v.contains(o.section) || !v.at(o.section).contains(o.name)) continue;
        Json value = v.at(o.section).at(o.name);
        if (o.is_switch) value = "on";
        out_engine[o.section][o.name] = value;
    }
    return out_engine;
}

Json helpers_of(const Resolved& r) {
    Json out = Json::array();
    const Json& ph = r.manifest.at("process_helpers");
    for (size_t i = 0; i < ph.size(); ++i) {
        const Json& h = ph[i];
        out.push_back({{"role", "helper" + std::to_string(i)}, {"gpu", h.at("bdf")}, {"vram_budget_mib", h.at("budget_mib")},
                       {"plugin_file", h.at("plugin_file")}, {"formats", h.at("formats")}, {"rows_kernel", h.at("rows_kernel")},
                       {"tier", h.at("placement")}, {"share", h.at("share")}});
    }
    return out;
}

Json static_provenance(bool measured, bool engine_only) {
    Json p = Json::object();
    p[""] = record("schema_default", "serving contract v1", "policy", "exact");
    p["/spec/intent/model"] = record("site", "site.toml:engine.model", "policy", "exact");
    p["/spec/intent/layout"] = record("profile", "layout selection (layouts.toml)", "policy", "exact");
    p["/spec/intent/service"] = record("profile", "serving profile (bc-serve --profile)", "policy", "exact");
    p["/spec/intent/power"] = record("cli", "bc-serve --power-profile", "policy", "exact");
    p["/spec/facts"] = record("detected", "bc-serve detect: kfd+sysfs+proc+statvfs+cgroup", "fact", "exact");
    p["/spec/calibration"] = record("calibrated", "calibration seeds and cache selected by layout and objective", "tuned", "measured",
                                    Json::array({"/spec/intent/layout", "/spec/intent/service/objective"}));
    Json resource_inputs = Json::array({"/spec/facts/host", "/spec/calibration"});
    if (measured) resource_inputs.push_back("/spec/plan/engine_memory/expected");
    p["/spec/resources"] = record("derived", measured ? "RSC03 host ledger over the measured plan" : "resolver memory settings (no measured plan yet)",
                                  "derived", "estimate", resource_inputs);
    if (!engine_only) {
    p["/spec/handoff/transport/shm"] = record("derived", "launcher shared-memory name", "derived", "exact");
    p["/spec/handoff/transport"] = record("profile", "flashnext launcher transport literals (interim hand-picked tunables; RSC08: derive or calibrate)", "derived", "estimate");
    p["/spec/handoff/placement"] = record("derived", "layout and heat policy", "derived", "exact", Json::array({"/spec/intent/layout"}));
    p["/spec/handoff/telemetry"] = record("site", "site.toml:handoff", "policy", "exact");
    p["/spec/handoff/kernels"] = record("profile", "flashnext launcher kernel literals (interim hand-picked tunables; RSC08: derive or calibrate)", "derived", "estimate");
    }
    p["/spec/processes/engine/device"] = record("derived", "layout engine card", "derived", "exact", Json::array({"/spec/facts/gpus", "/spec/intent/layout"}));
    p["/spec/processes/engine/runtime"] = record("site", "site.toml:engine runtime; architecture catalogue", "policy", "exact");
    p["/spec/processes/engine/server"] = record("site", "site.toml:engine port", "policy", "exact");
    p["/spec/processes/engine/scheduler"] = record("profile", "bc-serve resolver defaults and serving intent (interim)", "derived", "estimate", Json::array({"/spec/intent/service"}));
    p["/spec/processes/engine/execution"] = record("profile", "serving profile", "policy", "exact");
    if (!engine_only)
        p["/spec/processes/helper_common"] = record("profile", "serving/helper_profile.toml (parity-tested against the launcher and component settings)", "policy", "exact");
    p["/spec/processes/helpers"] = record("derived", "archs.toml, layout and budget derivation", "derived", "estimate",
                                          Json::array({"/spec/facts/gpus", "/spec/intent/layout", "/spec/calibration"}));
    p["/spec/plan"] = record("derived", "bc::experts owner selection and the rad0011 measured plan", "derived", "estimate",
                             Json::array({"/spec/intent", "/spec/facts", "/spec/calibration"}));
    return p;
}

/* resolver key -> the manifest leaf it becomes. Only these leaves can carry a resolved origin of their own. */
const std::map<std::string, std::string>& leaf_of_key() {
    static std::map<std::string, std::string> m = {
        {"engine.model", "/spec/intent/model"}, {"layout.name", "/spec/intent/layout"},
        {"service.objective", "/spec/intent/service/objective"}, {"service.max_sessions", "/spec/intent/service/max_sessions"},
        {"service.max_context", "/spec/intent/service/max_context_tokens"}, {"service.retention", "/spec/intent/service/retention"},
        {"service.quality", "/spec/intent/service/quality"},
        {"engine.radiance_home", "/spec/processes/engine/runtime/radiance_home"}, {"engine.kernels", "/spec/processes/engine/runtime/kernels"},
        {"engine.port", "/spec/processes/engine/server/port"}, {"engine.tp", "/spec/processes/engine/scheduler/tp"},
        {"engine.max_num_batched_tokens", "/spec/processes/engine/scheduler/max_num_batched_tokens"},
        {"engine.checkpoint_interval", "/spec/processes/engine/scheduler/checkpoint_interval"},
        {"engine.placement", "/spec/processes/engine/execution/placement"}, {"engine.ngram_placement", "/spec/processes/engine/execution/ngram_placement"},
        {"engine.kv_cache_dtype", "/spec/processes/engine/execution/kv_cache_dtype"},
        {"engine.num_speculative_tokens", "/spec/processes/engine/execution/num_speculative_tokens"},
        {"engine.rope_scaling", "/spec/processes/engine/execution/rope_scaling"}, {"engine.prefix_resume", "/spec/processes/engine/execution/prefix_resume"},
        {"engine.tp_wire", "/spec/processes/engine/execution/tp_wire"},
        {"engine.no_prefix_cache", "/spec/processes/engine/execution/no_prefix_cache"},
        {"engine.gpu_headroom_mib", "/spec/processes/engine/device/headroom_mib"},
        {"memory.host_pool_mib", "/spec/resources/host/pool_mib"}, {"memory.host_kv_mib", "/spec/resources/host/kv_mib"},
        {"memory.host_reserve_mib", "/spec/resources/host/reserve_mib"},
        {"handoff.shm", "/spec/handoff/transport/shm"}, {"handoff.max_rows", "/spec/handoff/transport/max_rows"},
        {"handoff.small_rows", "/spec/handoff/transport/small_rows"}, {"handoff.gemv_large_rows", "/spec/handoff/transport/gemv_large_rows"},
        {"handoff.skip_owned", "/spec/handoff/transport/skip_owned"}, {"handoff.moe_down_tile", "/spec/handoff/kernels/moe_down_tile"},
        {"handoff.histogram", "/spec/handoff/telemetry/histogram_file"}, {"handoff.histogram_period_s", "/spec/handoff/telemetry/period_s"},
        {"handoff.placement", "/spec/handoff/placement/strategy"}, {"handoff.heat", "/spec/handoff/placement/heat_file"},
        {"helper.combine", "/spec/processes/helper_common/combine/enabled"},
        {"helper.combine_min_rows", "/spec/processes/helper_common/combine/min_rows"},
        {"helper.combine_threads", "/spec/processes/helper_common/combine/threads"},
        {"helper.power_save", "/spec/processes/helper_common/runtime/power_save"},
    };
    static const bool extended = [] {
        for (const auto& o : engine_option_fields()) m.emplace(o.key, o.path);
        return true;
    }();
    (void)extended;
    return m;
}

/* The record at the nearest ancestor of `pointer` in `p` (the class and confidence its leaves already have). */
const Json* covering(const Json& p, std::string pointer) {
    while (true) {
        if (p.contains(pointer)) return &p.at(pointer);
        if (pointer.empty()) return nullptr;
        pointer.erase(pointer.find_last_of('/'));
    }
}

/* Records for the manifest: the static records say how each section is produced; a leaf the operator set (--set, the site
 * file, the serving profile) gets its own record from Resolved::provenance, so an override is never filed under the
 * section's default origin. Class stays the class the leaf has at its position (the covering record's). */
Json provenance_of(const ManifestInputs& in, const Json& derived_policy_provenance, bool engine_only) {
    const Resolved& r = in.resolved;
    Json p = static_provenance(in.measured.has_value(), engine_only);
    if (!in.live_policy.is_null())
        p["/spec/handoff/placement/live_policy"] =
            record("cli", in.live_policy_from.c_str(), "policy", "exact", Json::array({"/spec/intent/layout"}));
    for (auto it = derived_policy_provenance.begin(); it != derived_policy_provenance.end(); ++it)
        p[it.key()] = it.value();
    if (!in.experiment.is_null())
        p["/spec/experiment"] = record("experiment", in.experiment_from.c_str(), "diagnostic", "exact",
                                       Json::array({engine_only ? "/spec/intent/layout" : "/spec/processes/helper_common"}));
    for (const auto& [key, pointer] : leaf_of_key()) {
        if (!r.provenance.contains(key)) continue;
        if (engine_only && (pointer.rfind("/spec/handoff", 0) == 0 || pointer.rfind("/spec/processes/helper_common", 0) == 0)) continue;
        const Json& rp = r.provenance.at(key);
        const std::string origin = rp.at("origin").get<std::string>();
        if (origin != "cli" && origin != "site" && origin != "profile") continue;
        const Json* base = covering(p, pointer);
        if (!base) continue;
        p[pointer] = record(origin.c_str(), rp.at("source").get<std::string>().c_str(), base->at("class").get<std::string>().c_str(),
                            "exact");
    }
    return p;
}

v2::Manifest checked_decode(const Json& doc, const char* step) {
    v2::Manifest m;
    if (const v2::Status e = v2::decode(doc, "", &m))
        fail(std::string(step) + ": " + (e->path.empty() ? std::string() : e->path + ": ") + e->message);
    return m;
}

std::string sha_hex_of_zero() { return std::string(64, '0'); }
}  // namespace

nlohmann::json helper_common_settings(const nlohmann::json& values) { return helper_common(values); }

std::string new_instance_id() {
    std::string id = slurp("/proc/sys/kernel/random/uuid");
    while (!id.empty() && (id.back() == '\n' || id.back() == ' ')) id.pop_back();
    if (id.size() != 36) fail("the kernel gave no UUID");
    return id;
}

std::string file_sha256(const fs::path& file) { return sha256_hex(slurp(file)); }

BuiltManifest build_manifest(const ManifestInputs& in) {
    const Json& values = in.resolved.values;
    const Json calibration = calibration_entries(in.resolved, in.calibration_cache);
    const std::string model = values.at("engine").at("model");
    if (!in.live_policy.is_null() && in.profile != "lab")
        fail("--live-policy is an explicit lab-only override; production derives from calibration");
    const bool engine_only = in.resolved.manifest.at("process_helpers").empty();   // contract amendment 12
    // The expert container is read only to place experts on helpers. An engine-only configuration places none, so a dense
    // model (no routed experts, which Container::open refuses) is a valid engine-only model.
    bc::experts::Container container;
    if (!engine_only && !container.open(model.c_str())) fail("cannot open the model container " + model);
    if (engine_only && (!in.live_policy.is_null() || (!in.experiment.is_null() && in.experiment.contains("shape"))))
        fail("an engine-only configuration has no helpers: no live placement policy or helper shape applies");
    Json live_policy = in.live_policy;
    Json derived_provenance = Json::object();
    if (live_policy.is_null() && in.profile == "production" && !engine_only) {
        std::string skipped_reason;
        if (const auto derived = derive_manifest_live_policy(in, container, &skipped_reason)) {
            live_policy = derived->policy;
            derived_provenance = derived->provenance;
        } else if (!skipped_reason.empty()) {
            std::fprintf(stderr, "live policy calibration skipped: %s; serving without live_policy\n",
                         skipped_reason.c_str());
            // The absent live_policy has no JSON pointer. Record the decision
            // on its existing placement parent, which survives strict reading.
            derived_provenance["/spec/handoff/placement"] =
                record("derived", ("live policy disabled: " + skipped_reason).c_str(),
                       "derived", "measured", Json::array({"/spec/calibration"}));
        }
    }


    Json spec = {{"intent",
                  {{"model", model}, {"layout", values.at("layout").at("name")},
                   {"power", {{"profile", in.s10.power_profile}, {"role_overrides", Json::object()}}},
                   {"service",
                    {{"profile", in.profile}, {"objective", values.at("service").at("objective")},
                     {"max_sessions", values.at("service").at("max_sessions")},
                     {"max_context_tokens", values.at("service").at("max_context")},
                     {"retention", values.at("service").at("retention")}, {"quality", values.at("service").at("quality")}}}}},
                 {"facts", facts_of(in.resolved.facts)},
                 {"calibration", calibration},
                 {"resources", {{"host", resources_of(in, calibration)}}},
                 {"processes", {{"engine", engine_of(values)}, {"helpers", helpers_of(in.resolved)}}},
                 {"plan", {{"state", in.measured ? "measured" : "unmeasured"},
                           // S10 replaces this draft plan with the resolved one before the document is written.
                           {"power", {{"mode", "observe"}, {"profile", in.s10.power_profile},
                                      {"profile_digest", "sha256:" + sha_hex_of_zero()}, {"devices", draft_power_devices(in.resolved)},
                                      {"cpu", {{"settings", Json::object()}}}}},
                           {"owner_map", {{"owner_map_sha256", sha_hex_of_zero()}, {"helpers", Json::array()}}}}}};
    if (in.measured) {
        if (!in.s10.p1_record.is_object() || !in.s10.p1_record.contains("resource_inputs"))
            fail("a measured manifest needs the native P1 measurement record (--p1-record)");
        spec["plan"]["resource_inputs"] = in.s10.p1_record.at("resource_inputs");
    }
    if (!engine_only) {
        spec["handoff"] = handoff_of(values, live_policy);
        // S10 decides the live policy at startup (live_policy_state); the draft carries a placeholder decision.
        spec["handoff"]["placement"]["live_policy_state"] = {
            {"requested", "off"}, {"configured", false}, {"reason_code", "draft"}, {"reason_detail", "draft"},
            {"calibration_key", ""}, {"record_identity", ""}, {"effect_gate", false}, {"histogram_period_s", 0},
            {"cadence_ns", 0}, {"max_heat_age_ns", 0}};
        spec["processes"]["helper_common"] = helper_common(values);
    }
    if (in.measured)
        spec["plan"]["engine_memory"] = {{"cache_version", 1}, {"cache_key", sha_hex_of_zero()},
                                         {"model_identity", in.measured->key_inputs.at("model")},
                                         {"engine_identity", in.measured->key_inputs.at("engine_build")},
                                         {"expected", in.measured->plan.plan_json}};
    if (!in.experiment.is_null()) {
        if (in.profile != "lab") fail("an experiment needs the lab profile");
        spec["experiment"] = in.experiment;
    }
    Json doc = {{"apiVersion", v2::kApiVersion}, {"kind", v2::kKind},
                {"metadata",
                 {{"instance_id", in.instance_id}, {"producer", {{"name", "bc-serve"}, {"build_sha256", in.producer_sha256}}},
                  {"contract_version", v2::kContractVersion}, {"content_digest", "sha256:" + sha_hex_of_zero()}}},
                {"spec", std::move(spec)}, {"provenance", provenance_of(in, derived_provenance, engine_only)}};

    // The owner map is the planner's answer for the placement this very document declares: decode a draft with an empty
    // map, ask the shared planner (the policies come from the draft), write its sets and digests into the document.
    if (engine_only) {
        doc["spec"]["plan"]["owner_map"] = {{"owner_map_sha256", sha256_hex("")}, {"helpers", Json::array()}};
    } else {
        const std::size_t helper_count = doc.at("spec").at("processes").at("helpers").size();
        Json draft = doc;
        Json empty = Json::array();
        for (std::size_t i = 0; i < helper_count; ++i)
            empty.push_back({{"owned_sha256", sha_hex_of_zero()}, {"count", 0}, {"bytes", 0}, {"experts", Json::array()}});
        draft["spec"]["plan"]["owner_map"]["helpers"] = empty;
        const v2::Manifest m = checked_decode(draft, "draft manifest");
        // A derived budget (RAD49) was sized for the helper_common lanes and power save (worker_footprint.h). An experiment
        // that patches either for a helper would make that helper's lanes differ from the footprint its budget leaves room for,
        // so it needs the layout's explicit budgets.
        if (in.resolved.provenance.contains("memory.helper_budgets_mib") &&
            in.resolved.provenance.at("memory.helper_budgets_mib").at("origin") == "derived" && m.spec.processes.helper_common)
            for (std::size_t i = 0; i < helper_count; ++i) {
                const auto eff = v2::effective_helper_settings(m, i);
                const auto& common = *m.spec.processes.helper_common;
                if (eff.lanes.small_lanes != common.lanes.small_lanes || eff.runtime.power_save != common.runtime.power_save)
                    fail("helper " + std::to_string(i) + ": an experiment changes its small lanes or power save, but its budget "
                         "is derived from the helper_common lanes; give the layout's helpers budget-mib for such an experiment");
            }
        bc::experts::OwnerMap planned;
        std::string why;
        if (!bc::experts::plan_owners(container, owner_policies(m), helper_archs(m), helper_budgets(m), &planned, &why))
            fail("the declared placement cannot be planned for this model: " + why);
        Json helpers = Json::array();
        std::string all;
        for (std::size_t i = 0; i < planned.helpers.size(); ++i) {
            Json experts = Json::array();
            std::vector<std::array<int64_t, 2>> pairs;
            for (auto [l, e] : planned.helpers[i].expert) {
                experts.push_back(Json::array({l, e}));
                pairs.push_back({static_cast<int64_t>(l), static_cast<int64_t>(e)});
            }
            const std::string segment = v2::owner_segment(i, pairs);
            all += segment;
            helpers.push_back({{"owned_sha256", sha256_hex(segment)}, {"count", experts.size()},
                               {"bytes", planned.helpers[i].bytes}, {"experts", experts}});
        }
        doc["spec"]["plan"]["owner_map"] = {{"owner_map_sha256", sha256_hex(all)}, {"helpers", helpers}};
    }
    // The plan cache key follows from the document (spec section 5.1), so it is computed last but one; the digest last.
    if (in.measured)
        doc["spec"]["plan"]["engine_memory"]["cache_key"] = v2::plan_cache_key(checked_decode(doc, "manifest with its owner map"));
    // S10: the power plan, the S8 resource ledger and handoff (measured manifests), the live-policy startup decision.
    const v2::Manifest draft = checked_decode(doc, "manifest draft");
    auto s10_in = v2::s10_inputs(draft, in.resolved, in.s10);
    if (!s10_in) fail(v2::render_human("bc-serve", s10_in.error()));
    auto resolved = v2::resolve_s10(draft, s10_in.value());
    if (!resolved) fail(v2::render_human("bc-serve", resolved.error()));
    // The plan cache key follows from the document (spec section 5.1): S10 changed the budgets and the host ledger it covers.
    if (resolved.value().spec.plan.engine_memory)
        resolved.value().spec.plan.engine_memory->cache_key = v2::plan_cache_key(resolved.value());
    auto written = v2::write_s10(resolved.value());
    if (!written) fail(v2::render_human("bc-serve", written.error()));
    BuiltManifest out;
    out.document = Json::parse(written.value());
    out.text = out.document.dump(2) + "\n";
    out.digest = out.document.at("metadata").at("content_digest").get<std::string>();
    // The gate: what a consumer will do with these bytes.
    v2::Result<v2::Manifest> back = v2::read_manifest(out.text, "the manifest bc-serve built",
                                                      in.measured ? v2::PlanUse::serving : v2::PlanUse::measuring);
    if (!back.ok()) fail(v2::render_human("bc-serve", back.error()));
    if (const v2::Status e = verify_owner_map(back.value(), container)) fail(v2::render_human("bc-serve", *e));
    return out;
}

void write_manifest_file(const fs::path& path, const std::string& text) {
    // A manifest is immutable and its digest names it: never replace an existing one, even when two writers race.
    // Write a unique temporary file in the same directory, make it durable, then link it into place without replacing
    // (renameat2 RENAME_NOREPLACE fails with EEXIST if the target appeared since we started).
    const fs::path dir = path.has_parent_path() ? path.parent_path() : fs::path(".");
    std::string tmpl = (dir / (path.filename().string() + ".tmp.XXXXXX")).string();
    const int fd = ::mkstemp(tmpl.data());
    if (fd < 0) fail("cannot create a temporary file beside " + path.string() + ": " + std::strerror(errno));
    auto cleanup = [&] { ::unlink(tmpl.c_str()); };
    const char* p = text.data();
    size_t left = text.size();
    while (left > 0) {
        const ssize_t n = ::write(fd, p, left);
        if (n < 0) {
            if (errno == EINTR) continue;
            const int e = errno; ::close(fd); cleanup();
            fail("cannot write " + tmpl + ": " + std::strerror(e));
        }
        p += n; left -= static_cast<size_t>(n);
    }
    if (::fchmod(fd, 0444) != 0 || ::fsync(fd) != 0) {
        const int e = errno; ::close(fd); cleanup();
        fail("cannot make " + tmpl + " durable: " + std::strerror(e));
    }
    ::close(fd);
    if (::syscall(SYS_renameat2, AT_FDCWD, tmpl.c_str(), AT_FDCWD, path.c_str(), RENAME_NOREPLACE) != 0) {
        const int e = errno; cleanup();
        if (e == EEXIST) fail("refusing to replace " + path.string() + ": a manifest is immutable");
        fail("cannot place " + path.string() + ": " + std::strerror(e));
    }
    const int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (dfd >= 0) { ::fsync(dfd); ::close(dfd); }
}

}  // namespace bc::serving
