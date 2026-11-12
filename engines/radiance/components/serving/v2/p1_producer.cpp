#include "v2/p1_producer.h"

#include "experts.h"
#include "manifest_writer.h"
#include "owner_map.h"
#include "plan_cache.h"
#include "sha256.h"
#include "serving_codec.h"
#include "generated/v2/serving_codec.h"
#include "v2/p1_live_room.h"
#include "v2/p1_measurement.h"
#include "worker_footprint.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace bc::serving::v2 {
namespace {
using json = nlohmann::json;
namespace fs = std::filesystem;
constexpr int64_t kMiB = int64_t{1} << 20;
constexpr int64_t kMax = (int64_t{1} << 53) - 1;

[[noreturn]] void refuse(const std::string& field, const std::string& why) {
  throw std::runtime_error("P1 native " + field + ": " + why);
}
json read_strict(const fs::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) refuse(file.string(), "missing measuring-start artifact");
  const std::string bytes((std::istreambuf_iterator<char>(in)), {});
  return bc::serving::parse_plan_json(bytes, file.string());
}
int64_t integral(const json& j, const char* name, bool positive = false) {
  if (!j.is_number_integer()) refuse(name, "exact integer required");
  const int64_t n = j.get<int64_t>();
  if (n < 0 || n > kMax || (positive && !n))
    refuse(name, "out of measured wire range");
  return n;
}
void fields(const json& j, std::initializer_list<const char*> names, const char* where) {
  if (!j.is_object() || j.size() != names.size()) refuse(where, "missing or unknown members");
  for (const char* name : names)
    if (!j.contains(name)) refuse(where, std::string("missing ") + name);
}
std::string nonempty(const json& j, const char* name) {
  if (!j.is_string() || j.get<std::string>().empty())
    refuse(name, "nonempty native source required");
  return j.get<std::string>();
}
std::string file_sha(const fs::path& file) {
  if (!fs::is_regular_file(file)) refuse(file.string(), "source binary is not a regular file");
  return bc::serving::file_sha256(file);
}
void check_margin_and_run(const json& request) {
  fields(request.at("margin"), {"key", "source", "evidence", "mib"}, "margin");
  const auto& margin = request.at("margin");
  if (margin.at("key") != "host_ledger_margin.mib" || (margin.at("source") != "cache" && margin.at("source") != "seed"))
    refuse("margin", "the host-ledger margin is a seed or a calibration-cache value");
  (void)nonempty(margin.at("evidence"), "margin.evidence");
  (void)integral(margin.at("mib"), "margin.mib");
  (void)nonempty(request.at("run_id"), "run_id");
}
} // namespace

json produce_p1_inputs(const bc::serving::Resolved& resolved, const json& facts,
                       const fs::path& sys_root, const json& request) {
  fields(request, {"engine_plan_file", "rank_demand_file", "engine_binary",
                   "helper_binary", "retention_target_tokens", "margin", "run_id"}, "request");
  check_margin_and_run(request);
  json sources = request;
  sources["engine_plan"] = read_strict(nonempty(request.at("engine_plan_file"), "engine_plan_file"));
  sources["rank_demand"] = read_strict(nonempty(request.at("rank_demand_file"), "rank_demand_file"));
  sources.erase("engine_plan_file");
  sources.erase("rank_demand_file");
  return produce_p1(resolved, facts, sys_root, sources);
}

json produce_p1(const bc::serving::Resolved& resolved, const json& facts,
                const fs::path& sys_root, const json& request) {
  fields(request, {"engine_plan", "rank_demand", "engine_binary",
                   "helper_binary", "retention_target_tokens", "margin", "run_id"}, "sources");
  const auto& margin = request.at("margin");
  check_margin_and_run(request);
  const std::string evidence = margin.at("evidence").get<std::string>();
  const int64_t margin_mib = margin.at("mib").get<int64_t>();
  const std::string run_id = request.at("run_id").get<std::string>();
  const auto& selected = resolved.typed.layout;
  const auto& service = resolved.typed.service;
  if (selected.engine_cards.empty() || !service.max_context || !service.max_sessions)
    refuse("layout", "selected engine ranks and service targets required");

  const fs::path model_path = resolved.typed.engine.model;
  // The routed experts are read only to place them on helpers and to count what the engine keeps of them. A configuration
  // with no helpers places none, so a dense model (no routed experts, which Container::open refuses) is a valid engine-only
  // model, as for the manifest writer (contract amendment 12).
  const bool engine_only = selected.helper_cards.empty();
  bc::experts::Container model;
  const bool routed = model.open(model_path.c_str()) && model.layers && model.experts;
  if (!routed && !engine_only)
    refuse("model", "model container must be readable and have routed layers");
  if (!fs::is_regular_file(model_path)) refuse("model", "model container must be readable");
  const std::string model_id = bc::serving::container_identity(model_path);
  const fs::path engine_binary = nonempty(request.at("engine_binary"), "engine_binary");
  const std::string engine_id = bc::serving::engine_identity(
      engine_binary, resolved.typed.engine.radiance_home);
  const std::string helper_path = request.at("helper_binary").get<std::string>();
  if (helper_path.empty() != selected.helper_cards.empty())
    refuse("helper_binary", "helper binary required for selected helpers, absent otherwise");
  const std::string helper_id = selected.helper_cards.empty()
      ? bc::serving::sha256_hex(json::array().dump())
      : file_sha(helper_path);

  bc::serving::ManifestInputs inputs{};
  inputs.resolved = resolved;
  inputs.profile = "production";
  inputs.instance_id = bc::serving::new_instance_id();
  inputs.producer_sha256 = file_sha("/proc/self/exe");
  inputs.s10.power_profile = "stock";   // only the owner plan is read from this manifest
  const bc::serving::BuiltManifest owned = bc::serving::build_manifest(inputs);
  const auto checked = bc::serving::v2::read_manifest(
      owned.text, "native P1 owner-plan", bc::serving::v2::PlanUse::measuring);
  if (!checked.ok()) refuse("owner_map", "actual selected owner-plan failed strict validation");
  const auto& manifest = checked.value();
  P1Native native{};
  const auto& source_map = owned.document.at("spec").at("plan").at("owner_map");
  if (const auto error = decode(source_map, "/spec/plan/owner_map", &native.owner_map))
    refuse("owner_map", error->message);
  // build_manifest ran and verified the exact shared owner planner against
  // the actual .rad container before the v2 owner-map decoder was called.
  if (native.owner_map.helpers.size() != selected.helper_cards.size())
    refuse("owner_map", "verified ownership differs from selected layout");
  const int64_t ctx = service.max_context;
  native.context_target_tokens = ctx;
  native.concurrent_sessions = service.max_sessions;
  // 0: no host KV tier (an engine-only layout without one); retention is then nothing to admit (amendment 19)
  native.retention_target_tokens = integral(
      request.at("retention_target_tokens"), "retention_target_tokens", false);
  if (native.retention_target_tokens > ctx * native.concurrent_sessions)
    refuse("retention_target_tokens", "exceeds measured context times concurrency");

  const json& plan = request.at("engine_plan");
  if (const auto e = decode(plan, "rad0011 engine plan", &native.engine_plan))
    refuse("engine_plan", e->message);
  const json& ranks = request.at("rank_demand");
  fields(ranks, {"schema_version", "kind", "gpu", "resident_other_bytes"}, "rank demand");
  if (ranks.at("schema_version") != 1 || ranks.at("kind") != "radiance_p1_rank_demand" ||
      !ranks.at("gpu").is_array() || ranks.at("gpu").size() != selected.engine_cards.size())
    refuse("rank demand", "invalid source identity or rank count");
  native.resident_other_bytes = integral(ranks.at("resident_other_bytes"), "resident_other_bytes");
  for (size_t i = 0; i < selected.engine_cards.size(); ++i) {
    const auto& g = ranks.at("gpu").at(i);
    fields(g, {"bdf", "required_bytes", "fixed_bytes", "claimable_bytes", "evidence"}, "GPU");
    if (g.at("bdf") != selected.engine_cards[i] ||
        nonempty(g.at("evidence"), "GPU evidence").find("rad0011") == std::string::npos)
      refuse("GPU", "post-rank-split BDF/source mismatch");
    const int64_t required = integral(g.at("required_bytes"), "GPU required", true);
    const int64_t fixed = integral(g.at("fixed_bytes"), "GPU fixed", true);
    const int64_t claimable = integral(g.at("claimable_bytes"), "GPU claimable", true);
    if (fixed > required || required > claimable)
      refuse("GPU", "native fixed/required demand exceeds measured claimable room");
    native.gpu.push_back({selected.engine_cards[i], required, fixed, claimable});
  }
  if (native.engine_plan.vram.cards.size() != native.gpu.size())
    refuse("GPU", "engine plan rank count differs from demand sidecar");

  std::set<std::pair<uint32_t,uint32_t>> used;
  for (const auto& helper : native.owner_map.helpers)
    for (const auto& expert : helper.experts) {
      if (expert[0] < 0 || expert[1] < 0 ||
          expert[0] >= model.layers || expert[1] >= model.experts)
        refuse("owner_map", "expert index outside actual model");
      if (!used.insert({uint32_t(expert[0]),uint32_t(expert[1])}).second)
        refuse("owner_map", "expert owned by more than one helper");
    }
  int64_t unowned_bytes = 0;
  for (uint32_t l = 0; routed && l < model.layers; ++l)
    for (uint32_t e = 0; e < model.experts; ++e) {
      const auto key = std::make_pair(l,e);
      const bool gate = model.weight(l,e,0) != nullptr;
      const bool down = model.weight(l,e,1) != nullptr;
      if (!gate && !down) continue;
      if (gate != down) refuse("model", "expert missing a weight slot");
      const int64_t count = bc::experts::canonical_expert_bytes(model,l,e);
      if (count <= 0) refuse("model", "expert missing complete plane sizes");
      if (!used.count(key)) {
        if (count > kMax - unowned_bytes) refuse("unowned_experts", "overflow");
        unowned_bytes += count;
      }
    }
  native.unowned_expert_bytes = unowned_bytes;
  namespace fp = r3::worker_footprint;
  const fp::Shape shape{model.layers, model.experts, model.hidden, model.intermediate,
                        (model.top_k ? model.top_k : r3::moe_shm::kTopK), r3::moe_shm::kActGroup};
  const uint32_t rows = resolved.typed.handoff.max_rows;
  const uint32_t small = resolved.typed.handoff.small_rows;
  for (size_t i = 0; i < native.owner_map.helpers.size(); ++i) {
    const auto effective = bc::serving::v2::effective_helper_settings(manifest, i);
    std::vector<uint32_t> thresholds;
    for (const int64_t v : effective.lanes.small_lanes) {
      if (v <= 0 || v > std::numeric_limits<uint32_t>::max())
        refuse("helper.lanes", "invalid actual small lane threshold");
      thresholds.push_back(static_cast<uint32_t>(v));
    }
    const auto classes = fp::small_classes(thresholds, small, rows);
    const auto lanes = fp::lanes_for(classes, rows, native.owner_map.helpers.size(), i,
                                    selected.engine_cards.size(), effective.runtime.power_save);
    uint64_t largest = 0;
    for (const auto& pair : native.owner_map.helpers[i].experts) {
      const uint32_t l = uint32_t(pair[0]), e = uint32_t(pair[1]);
      const int64_t bytes = bc::experts::laid_out_bytes(model, true, l, e);
      if (bytes <= 0) refuse("helper.expert", "device layout has no measured size");
      largest = std::max(largest, uint64_t(bytes));
    }
    if (!largest) refuse("helper.expert", "selected helper owns no experts");
    native.helpers.push_back({"helper" + std::to_string(i), selected.helper_cards[i],
                              shape, lanes, largest});
  }
  std::vector<std::string> all = selected.engine_cards;
  all.insert(all.end(), selected.helper_cards.begin(), selected.helper_cards.end());
  const auto room = measure_live_room(facts,sys_root,all);
  if (!room) refuse("live host", room.error().path + ": " + room.error().message);
  native.mem_available_bytes = room.value().mem_available_bytes;
  native.cgroup_headroom_bytes = room.value().cgroup_headroom_bytes;
  native.pin_available_bytes = room.value().pin_available_bytes;
  if (resolved.typed.memory.host_reserve_mib > uint64_t(kMax / kMiB))
    refuse("host_reserve", "resolved host reserve overflows bytes");
  native.host_reserve_bytes = static_cast<int64_t>(resolved.typed.memory.host_reserve_mib) * kMiB;
  native.host_margin = {"host_ledger_margin.mib", margin.at("source") == "seed" ? MarginSource::kSeed : MarginSource::kCache, evidence, margin_mib};
  native.model_digest = model_id;
  native.engine_build_digest = engine_id;
  native.helper_build_digest = helper_id;
  native.layout_digest = bc::serving::sha256_hex(resolved.values.at("layout").dump());
  native.inventory_digest = bc::serving::sha256_hex(facts.at("gpus").dump());
  native.planning_key = bc::serving::sha256_hex(
      model_id + engine_id + helper_id + native.layout_digest + native.inventory_digest +
      native.owner_map.owner_map_sha256 + std::to_string(ctx) +
      std::to_string(native.retention_target_tokens) + ranks.dump() + plan.dump());
  native.provenance = run_id + "; " + room.value().evidence +
      "; verified container identity and owner planner; rank split";
  const auto published = publish_p1_measurement(native);
  if (!published) refuse(published.error().path, published.error().message);
  return published.value();
}
} // namespace bc::serving::v2
