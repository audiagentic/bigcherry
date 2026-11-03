#include "manifest.h"

#include "jcs.h"
#include "serving_codec.h"
#include "sha256.h"
#include "strict_json.h"

#include <fstream>
#include <functional>
#include <map>
#include <regex>
#include <set>
#include <sstream>

namespace bc::serving::v1 {
namespace {

using Json = nlohmann::json;
using C = ConfigErrorCode;
const char* const kHint = "regenerate the manifest with bc-serve; a v1 manifest is never edited by hand";

ConfigError err(C code, std::string path, std::string message, std::string hint = kHint) {
    return make_error(code, std::move(path), std::move(message), std::move(hint));
}

std::string helper_at(size_t i) { return "spec.processes.helpers[" + std::to_string(i) + "]"; }

/* The contract identity, before anything else is trusted: another version cannot be digested by v1's rules. */
Status identity(const Json& doc) {
    if (!doc.is_object()) return err(C::invalid_format, "", "a manifest is a JSON object");
    const auto api = doc.find("apiVersion");
    if (api == doc.end() || !api->is_string() || *api != kApiVersion)
        return err(C::unsupported_version, "apiVersion",
                   "apiVersion must be \"" + std::string(kApiVersion) + "\"" +
                       (api == doc.end() ? std::string(", is absent") : ", is " + api->dump()),
                   "this reader supports bigcherry/v1 only; resolve the manifest with the matching bc-serve");
    const auto kind = doc.find("kind");
    if (kind == doc.end() || !kind->is_string() || *kind != kKind)
        return err(C::unsupported_version, "kind", "kind must be \"" + std::string(kKind) + "\"");
    const auto meta = doc.find("metadata");
    if (meta != doc.end() && meta->is_object()) {
        const auto v = meta->find("contract_version");
        if (v != meta->end() && (!v->is_number_integer() || v->get<int64_t>() != kContractVersion))
            return err(C::unsupported_version, "metadata.contract_version",
                       "contract_version must be " + std::to_string(kContractVersion) + ", is " + v->dump());
    }
    return std::nullopt;
}

/* spec §8: sha256 over the JCS bytes of the manifest without metadata.content_digest. */
Status digest(const Json& doc) {
    const auto meta = doc.find("metadata");
    if (meta == doc.end() || !meta->is_object())
        return err(C::digest_mismatch, "metadata.content_digest", "the manifest has no metadata.content_digest");
    const auto d = meta->find("content_digest");
    if (d == meta->end()) return err(C::digest_mismatch, "metadata.content_digest", "metadata.content_digest is absent");
    if (!d->is_string()) return err(C::digest_mismatch, "metadata.content_digest", "metadata.content_digest must be a string");
    static const std::regex form("^sha256:[0-9a-f]{64}$");
    const std::string& recorded = d->get_ref<const std::string&>();
    if (!std::regex_match(recorded, form))
        return err(C::digest_mismatch, "metadata.content_digest",
                   "metadata.content_digest must be \"sha256:\" and 64 lowercase hex digits");
    std::string computed, why;
    if (!bc::serving::manifest_content_digest(doc, &computed, &why))
        return err(C::invalid_format, "", "the manifest cannot be canonicalized: " + why);
    if (computed != recorded)
        return err(C::digest_mismatch, "metadata.content_digest",
                   "content digest mismatch: recorded " + recorded + ", computed " + computed,
                   "the manifest changed after it was resolved; resolve it again with bc-serve");
    return std::nullopt;
}

/* spec §7: every provenance key and input names a value in the document, and inputs among records form no cycle. */
Status provenance(const Json& doc) {
    const Json& prov = doc.at("provenance");
    auto at = [](const std::string& key) { return "provenance[" + Json(key).dump() + "]"; };
    for (const auto& [key, record] : prov.items()) {
        if (!doc.contains(Json::json_pointer(key)))
            return err(C::provenance_invalid, at(key), "the record's pointer names no value in the manifest");
        for (const Json& in : record.at("inputs"))
            if (!doc.contains(Json::json_pointer(in.get<std::string>())))
                return err(C::provenance_invalid, at(key), "input " + in.dump() + " names no value in the manifest");
    }
    // Iterative depth-first search with explicit visiting / done states: a chain of dependencies as long as the
    // document allows (tens of thousands of records) must not recurse on the C++ stack.
    std::map<std::string, int> state;   // 1 visiting, 2 done
    for (const auto& [root, unused] : prov.items()) {
        if (state[root] != 0) continue;
        std::vector<std::pair<std::string, size_t>> stack{{root, 0}};
        state[root] = 1;
        while (!stack.empty()) {
            auto& [key, next] = stack.back();
            const Json& inputs = prov.at(key).at("inputs");
            if (next == inputs.size()) {
                state[key] = 2;
                stack.pop_back();
                continue;
            }
            const std::string k = inputs[next++].get<std::string>();
            if (!prov.contains(k)) continue;
            if (state[k] == 1) return err(C::provenance_invalid, at(key), "dependency cycle through " + Json(k).dump());
            if (state[k] == 0) {
                state[k] = 1;
                stack.emplace_back(k, 0);
            }
        }
    }
    return std::nullopt;
}

void identity_leaves(const Json& v, const std::string& pointer, std::vector<std::string>* out) {
    if (v.is_object() && !v.empty()) {
        for (const auto& [k, sub] : v.items()) {
            std::string token;
            for (char ch : k) token += ch == '~' ? std::string("~0") : ch == '/' ? std::string("~1") : std::string(1, ch);
            identity_leaves(sub, pointer + "/" + token, out);
        }
    } else {
        out->push_back(pointer);
    }
}

Status sorted_lanes(const std::vector<int64_t>& lanes, const std::string& at) {
    for (size_t i = 1; i < lanes.size(); ++i)
        if (lanes[i] <= lanes[i - 1])
            return err(C::invalid_value, at, "small_lanes must be strictly ascending");
    return std::nullopt;
}

HelperCommon effective(const Manifest& m, size_t i) {
    HelperCommon s = m.spec.processes.helper_common;
    const HelperProcess& h = m.spec.processes.helpers[i];
    if (h.settings) apply(s, *h.settings);
    if (m.spec.experiment && m.spec.experiment->shape) {
        const ExperimentShape& shape = *m.spec.experiment->shape;
        if (shape.helper_common) apply(s, *shape.helper_common);
        if (shape.helpers)
            for (const ShapePatch& p : *shape.helpers)
                if (p.role == h.role) apply(s, p.settings);
    }
    return s;
}

}  // namespace

HelperCommon effective_helper_settings(const Manifest& m, size_t helper) { return effective(m, helper); }

Status check_identity_coverage(const nlohmann::json& doc) {
    std::vector<std::string> leaves;
    for (const char* top : {"apiVersion", "kind", "metadata"})
        if (doc.contains(top)) identity_leaves(doc.at(top), std::string("/") + top, &leaves);
    const Json empty = Json::object();
    const Json& prov = doc.contains("provenance") && doc.at("provenance").is_object() ? doc.at("provenance") : empty;
    for (const std::string& leaf : leaves) {
        if (std::find(kIdentityExempt.begin(), kIdentityExempt.end(), leaf) != kIdentityExempt.end()) continue;
        bool covered = false;
        for (std::string p = leaf; !p.empty() && !covered; p = p.substr(0, p.rfind('/'))) covered = prov.contains(p);
        if (!covered)
            return err(C::provenance_invalid, "provenance",
                       "identity field " + leaf + " is not exempt (spec §11 amendment 6) and has no provenance record");
    }
    return std::nullopt;
}

std::string owner_segment(size_t helper, const std::vector<std::array<int64_t, 2>>& experts) {
    std::string s = "h" + std::to_string(helper) + "\n";
    for (const auto& [layer, expert] : experts) s += std::to_string(layer) + ":" + std::to_string(expert) + "\n";
    return s;
}

std::string plan_cache_key(const Manifest& m) {
    const Spec& s = m.spec;
    std::map<std::string, const GpuFact*> gpus;
    for (const GpuFact& g : s.facts.gpus) gpus[g.bdf] = &g;
    const EngineProcess& e = s.processes.engine;
    const GpuFact* card = e.device.gpus.empty() || !gpus.count(e.device.gpus[0]) ? nullptr : gpus.at(e.device.gpus[0]);
    Json helpers = Json::array();
    for (size_t i = 0; i < s.processes.helpers.size(); ++i) {
        const HelperProcess& h = s.processes.helpers[i];
        Json formats = Json::array();
        for (ExpertFormat f : h.formats) formats.push_back(kExpertFormatNames[static_cast<size_t>(f)]);
        helpers.push_back(Json{{"role", h.role}, {"arch", gpus.count(h.gpu) ? gpus.at(h.gpu)->arch : std::string()},
                               {"plugin_file", h.plugin_file}, {"formats", formats}, {"rows_kernel", h.rows_kernel},
                               {"settings", encode(effective(m, i))}});
    }
    const Service& sv = s.intent.service;
    const EngineExecution& x = e.execution;
    const Json inputs = {
        {"cache_version", 1},
        {"model", s.plan.engine_memory.model_identity},
        {"engine_build", s.plan.engine_memory.engine_identity},
        {"card", {{"arch", card ? card->arch : std::string()}, {"vram_total_mib", card ? card->vram.total_mib : 0}}},
        {"layout", s.intent.layout},
        {"owner_map_sha256", s.plan.owner_map.owner_map_sha256},
        {"planning_options",
         {{"objective", kObjectiveNames[static_cast<size_t>(sv.objective)]},
          {"retention", kRetentionNames[static_cast<size_t>(sv.retention)]},
          {"max_sessions", sv.max_sessions},
          {"max_context_tokens", sv.max_context_tokens},
          {"tp", e.scheduler.tp},
          {"max_num_batched_tokens", e.scheduler.max_num_batched_tokens},
          {"checkpoint_interval", e.scheduler.checkpoint_interval},
          {"placement", kWeightPlacementNames[static_cast<size_t>(x.placement)]},
          {"ngram_placement", kNgramPlacementNames[static_cast<size_t>(x.ngram_placement)]},
          {"kv_cache_dtype", kKvCacheDtypeNames[static_cast<size_t>(x.kv_cache_dtype)]},
          {"num_speculative_tokens", x.num_speculative_tokens},
          {"rope_scaling", x.rope_scaling},
          {"prefix_resume", kPrefixResumeNames[static_cast<size_t>(x.prefix_resume)]},
          {"kernels", e.runtime.kernels},
          {"handoff_max_rows", s.handoff.transport.max_rows},
          {"helpers", helpers}}}};
    std::string bytes, why;
    if (!bc::serving::jcs_canonicalize(inputs, &bytes, &why)) return "";
    return bc::serving::sha256_hex(bytes);
}

Status validate_semantics(const Manifest& m) {
    const Spec& s = m.spec;
    // §6: experiments are lab-only; a lab placement experiment's step follows its mode
    if (s.experiment && s.intent.service.profile == Profile::kProduction)
        return err(C::experiment_forbidden, "spec.experiment",
                   "spec.experiment is forbidden when spec.intent.service.profile=production",
                   "select profile=lab explicitly and regenerate the manifest");
    if (s.experiment && s.experiment->placement) {
        const ExperimentPlacement& p = *s.experiment->placement;
        if (p.mode == PlacementMode::kForcedChurn && !p.at_step)
            return err(C::missing_field, "spec.experiment.placement.at_step", "mode forced_churn requires at_step");
        if (p.mode == PlacementMode::kObserve && p.at_step)
            return err(C::invalid_value, "spec.experiment.placement.at_step", "mode observe takes no at_step");
    }
    // §2: the one receipt echo
    if (s.plan.engine_memory.expected.host_pool_mib != s.resources.host.pool_mib)
        return err(C::invalid_value, "spec.plan.engine_memory.expected.host_pool_mib",
                   "the measured plan's host_pool_mib " + std::to_string(s.plan.engine_memory.expected.host_pool_mib) +
                       " must equal spec.resources.host.pool_mib " + std::to_string(s.resources.host.pool_mib));
    // §5: the measured checkpoint floor is within the want
    const PlanCheckpoints& ck = s.plan.engine_memory.expected.checkpoints;
    if (ck.floor_slots > ck.want_slots)
        return err(C::invalid_value, "spec.plan.engine_memory.expected.checkpoints.floor_slots",
                   "floor_slots " + std::to_string(ck.floor_slots) + " exceed want_slots " + std::to_string(ck.want_slots));
    // calibration: one entry per key
    std::set<CalibrationKey> keys;
    for (size_t i = 0; i < s.calibration.size(); ++i)
        if (!keys.insert(s.calibration[i].key).second)
            return err(C::duplicate_field, "spec.calibration[" + std::to_string(i) + "].key",
                       "calibration key " + std::string(kCalibrationKeyNames[static_cast<size_t>(s.calibration[i].key)]) +
                           " is selected twice");
    // GPUs: every reference resolves, none is used twice
    std::set<std::string> detected;
    for (size_t i = 0; i < s.facts.gpus.size(); ++i)
        if (!detected.insert(s.facts.gpus[i].bdf).second)
            return err(C::duplicate_field, "spec.facts.gpus[" + std::to_string(i) + "].bdf",
                       "GPU " + s.facts.gpus[i].bdf + " is detected twice");
    std::set<std::string> used;
    const std::vector<std::string>& engine_gpus = s.processes.engine.device.gpus;
    for (size_t i = 0; i < engine_gpus.size(); ++i) {
        const std::string at = "spec.processes.engine.device.gpus[" + std::to_string(i) + "]";
        if (!detected.count(engine_gpus[i]))
            return err(C::unresolved_reference, at, "GPU " + engine_gpus[i] + " is not among the detected GPUs");
        if (!used.insert(engine_gpus[i]).second) return err(C::duplicate_field, at, "GPU " + engine_gpus[i] + " is repeated");
    }
    if (s.processes.engine.scheduler.tp != static_cast<int64_t>(engine_gpus.size()))
        return err(C::invalid_value, "spec.processes.engine.scheduler.tp",
                   "tp " + std::to_string(s.processes.engine.scheduler.tp) + " must equal the engine's " +
                       std::to_string(engine_gpus.size()) + " GPU(s)");
    // §2.2: helpers are helper0, helper1, ... on distinct detected GPUs
    const std::vector<HelperProcess>& helpers = s.processes.helpers;
    {
        std::set<std::string> roles;
        for (size_t i = 0; i < helpers.size(); ++i)
            if (!roles.insert(helpers[i].role).second)
                return err(C::duplicate_field, helper_at(i) + ".role", "role " + helpers[i].role + " is repeated");
    }
    for (size_t i = 0; i < helpers.size(); ++i) {
        if (helpers[i].role != "helper" + std::to_string(i))
            return err(C::invalid_value, helper_at(i) + ".role",
                       "helper " + std::to_string(i) + " must be role helper" + std::to_string(i) + ", is " +
                           helpers[i].role);
        if (!detected.count(helpers[i].gpu))
            return err(C::unresolved_reference, helper_at(i) + ".gpu",
                       "GPU " + helpers[i].gpu + " is not among the detected GPUs");
        if (!used.insert(helpers[i].gpu).second)
            return err(C::duplicate_field, helper_at(i) + ".gpu", "GPU " + helpers[i].gpu + " is already in use");
    }
    if (auto e = sorted_lanes(s.processes.helper_common.lanes.small_lanes, "spec.processes.helper_common.lanes.small_lanes"))
        return e;
    // lab shape: role patches name existing helpers, once each
    if (s.experiment && s.experiment->shape && s.experiment->shape->helpers) {
        std::set<std::string> seen;
        const auto& patches = *s.experiment->shape->helpers;
        for (size_t i = 0; i < patches.size(); ++i) {
            const std::string at = "spec.experiment.shape.helpers[" + std::to_string(i) + "].role";
            if (std::none_of(helpers.begin(), helpers.end(), [&](const HelperProcess& h) { return h.role == patches[i].role; }))
                return err(C::unresolved_reference, at, "no helper has role " + patches[i].role);
            if (!seen.insert(patches[i].role).second) return err(C::duplicate_field, at, "role " + patches[i].role + " is patched twice");
        }
    }
    // §2.1: the effective settings; coordination flags agree across helpers
    std::vector<HelperCommon> eff;
    for (size_t i = 0; i < helpers.size(); ++i) eff.push_back(effective(m, i));
    for (size_t i = 0; i < eff.size(); ++i) {
        const HelperPlacement& p = eff[i].placement;
        if (auto e = sorted_lanes(eff[i].lanes.small_lanes, helper_at(i) + " (effective) lanes.small_lanes")) return e;
        if (!p.balance && p.balance_speeds_milli)
            return err(C::invalid_value, helper_at(i) + " (effective) placement.balance_speeds_milli",
                       "balance_speeds_milli must be absent when balance is false");
        if (p.balance && (!p.balance_speeds_milli || p.balance_speeds_milli->size() != helpers.size()))
            return err(C::invalid_value, helper_at(i) + " (effective) placement.balance_speeds_milli",
                       "balance needs one speed per helper (" + std::to_string(helpers.size()) + ")");
        if (p.balance != eff[0].placement.balance || p.balance_speeds_milli != eff[0].placement.balance_speeds_milli ||
            eff[i].combine.enabled != eff[0].combine.enabled)
            return err(C::invalid_value, helper_at(i) + " (effective)",
                       "placement.balance, balance_speeds_milli and combine.enabled must be the same for every helper");
    }
    // §4: the initial owner map, aligned with the helpers, digests verified
    const OwnerMap& om = s.plan.owner_map;
    if (om.helpers.size() != helpers.size())
        return err(C::ownership_mismatch, "spec.plan.owner_map.helpers",
                   std::to_string(om.helpers.size()) + " owned set(s) for " + std::to_string(helpers.size()) + " helper(s)");
    std::set<std::array<int64_t, 2>> pairs;
    std::string all;
    for (size_t i = 0; i < om.helpers.size(); ++i) {
        const OwnedSet& o = om.helpers[i];
        const std::string at = "spec.plan.owner_map.helpers[" + std::to_string(i) + "]";
        if ((o.bytes == 0) != o.experts.empty())
            return err(C::ownership_mismatch, at + ".bytes",
                       std::to_string(o.experts.size()) + " owned expert(s) cannot take " + std::to_string(o.bytes) +
                           " device bytes");
        if (o.count != static_cast<int64_t>(o.experts.size()))
            return err(C::ownership_mismatch, at + ".count",
                       "count " + std::to_string(o.count) + " but " + std::to_string(o.experts.size()) + " owned experts");
        for (size_t k = 0; k < o.experts.size(); ++k)
            if (!pairs.insert(o.experts[k]).second)
                return err(C::ownership_mismatch, at + ".experts[" + std::to_string(k) + "]",
                           "expert [" + std::to_string(o.experts[k][0]) + ", " + std::to_string(o.experts[k][1]) +
                               "] is owned twice");
        const std::string segment = owner_segment(i, o.experts);
        if (bc::serving::sha256_hex(segment) != o.owned_sha256)
            return err(C::ownership_mismatch, at + ".owned_sha256", "owned_sha256 does not match the owned experts");
        all += segment;
    }
    if (bc::serving::sha256_hex(all) != om.owner_map_sha256)
        return err(C::ownership_mismatch, "spec.plan.owner_map.owner_map_sha256",
                   "owner_map_sha256 does not match the owned sets");
    // §5.1: the measured plan belongs to exactly this configuration
    const std::string key = plan_cache_key(m);
    if (key != s.plan.engine_memory.cache_key)
        return err(C::plan_mismatch, "spec.plan.engine_memory.cache_key",
                   "the measured plan's cache_key " + s.plan.engine_memory.cache_key +
                       " is not this configuration's " + key + " (spec §5.1)",
                   "measure the plan again (bc-serve --plan-memory) and resolve a new manifest");
    return std::nullopt;
}

Result<Manifest> read_manifest(std::string_view bytes, const std::string& source) {
    auto sourced = [&](ConfigError e) {
        if (!e.source_file) e.source_file = source;
        return e;
    };
    Result<Json> parsed = parse_strict_json(bytes, source);
    if (!parsed) return parsed.error();
    const Json& doc = parsed.value();
    if (auto e = identity(doc)) return sourced(*e);
    if (auto e = digest(doc)) return sourced(*e);
    // errors.md §6: production with any experiment key, even an empty one, is the experiment error, not a decode one
    if (const Json* sp = doc.contains("spec") ? &doc.at("spec") : nullptr; sp && sp->is_object() && sp->contains("experiment")) {
        const Json::json_pointer profile("/spec/intent/service/profile");
        if (doc.contains(profile) && doc.at(profile) == "production")
            return sourced(err(C::experiment_forbidden, "spec.experiment",
                               "spec.experiment is forbidden when spec.intent.service.profile=production",
                               "select profile=lab explicitly and regenerate the manifest"));
    }
    Manifest m;
    if (auto e = decode(doc, "", &m)) return sourced(*e);
    if (auto e = provenance(doc)) return sourced(*e);
    if (auto e = check_identity_coverage(doc)) return sourced(*e);
    if (auto e = validate_semantics(m)) return sourced(*e);
    return m;
}

Result<Manifest> load_manifest(const std::filesystem::path& path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) {
        ConfigError e = err(C::io_failure, "", "cannot read the manifest: " + ec.message(),
                            "check the path; bc-serve writes the manifest it starts the serving set with");
        e.source_file = path.string();
        return e;
    }
    if (size > kMaxDocumentBytes) {
        ConfigError e = err(C::invalid_format, "", "document is " + std::to_string(size) + " bytes, over the " +
                                                        std::to_string(kMaxDocumentBytes) + "-byte limit");
        e.source_file = path.string();
        return e;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        ConfigError e = err(C::io_failure, "", "cannot open the manifest");
        e.source_file = path.string();
        return e;
    }
    // at most one byte past the limit, however the file changes while it is read
    std::string text(kMaxDocumentBytes + 1, '\0');
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (in.bad()) {
        ConfigError e = err(C::io_failure, "", "cannot read the manifest");
        e.source_file = path.string();
        return e;
    }
    text.resize(static_cast<size_t>(in.gcount()));
    return read_manifest(text, path.string());
}

EngineView engine_view(const Manifest& m) {
    int64_t ppm = 0;
    for (const CalibrationEntry& c : m.spec.calibration)
        if (c.key == CalibrationKey::kKvBackingFloorFractionPpm) ppm = std::get<int64_t>(c.value);
    return EngineView{m.spec.intent, m.spec.resources, m.spec.handoff, m.spec.processes.engine, m.spec.plan.engine_memory,
                      ppm};
}

Result<HelperView> helper_view(const Manifest& m, std::string_view role) {
    const auto& helpers = m.spec.processes.helpers;
    for (size_t i = 0; i < helpers.size(); ++i)
        if (helpers[i].role == role) {
            HelperCommon settings = effective(m, i);
            const bool rows = helpers[i].rows_kernel || settings.lanes.rows;
            return HelperView{i, helpers[i], m.spec.plan.owner_map.helpers.at(i), m.spec.handoff, std::move(settings), rows};
        }
    return err(C::unresolved_reference, "spec.processes.helpers", "no helper has role " + std::string(role),
               "start the helper with a role the manifest defines");
}

}  // namespace bc::serving::v1
