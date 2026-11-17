#include "s8_replay.h"

#include "manifest_writer.h"
#include "sha256.h"
#include "v2/s8_rules.h"
#include "v2/strict_json.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace bc::serving::session {
namespace {
using Json = nlohmann::json;
using Code = v2::ConfigErrorCode;

v2::ConfigError refusal(std::string path, std::string reason) {
    return v2::make_error(Code::calibration_invalid, std::move(path), std::move(reason));
}

bool same_host(const v2::HostResources& a, const v2::HostResources& b) {
    return a.pool_mib == b.pool_mib && a.kv_mib == b.kv_mib && a.reserve_mib == b.reserve_mib &&
           a.checkpoint_slots == b.checkpoint_slots;
}

bool same_calibration(const v2::CalibrationEntry& a, const v2::CalibrationEntry& b) {
    return a.key == b.key && a.value == b.value && a.source == b.source && a.seeded == b.seeded &&
           a.evidence == b.evidence;
}

std::optional<v2::ConfigError> check_selected_evidence(const std::vector<v2::CalibrationEntry>& original,
                                                       const std::vector<v2::CalibrationEntry>& current,
                                                       const std::vector<v2::S8Calibration>& records,
                                                       int64_t now) {
    for (const auto& issued : original) {
        const auto it = std::find_if(current.begin(), current.end(),
                                     [&](const auto& row) { return row.key == issued.key; });
        if (it != current.end() && same_calibration(issued, *it))
            continue;
        // S8 can legitimately select the same fallback numerical seed as an
        // expired P2 record. Evidence identity is NOT numerically equivalent.
        for (const auto& record : records) {
            if (issued.evidence.find(record.evidence_id) != std::string::npos &&
                record.valid_until_unix_s <= now)
                return refusal("spec.calibration", "S8 record " + record.evidence_id +
                                                       " expired at valid_until_unix_s=" +
                                                       std::to_string(record.valid_until_unix_s));
        }
        return refusal("spec.calibration",
                       "S8 calibrated evidence changed or is no longer selected in current scope");
    }
    if (current.size() != original.size())
        return refusal("spec.calibration",
                       "S8 newly qualified records change the measured execution semantics");
    return std::nullopt;
}

v2::Status same_s8(const v2::Manifest& saved, const v2::S8Resolution& before, const v2::S8Resolution& after,
                   const std::vector<v2::S8Calibration>& records, int64_t now) {
    if (!same_host(before.host, saved.spec.resources.host) || !same_host(after.host, before.host))
        return refusal("spec.resources.host", "stored host pool/KV/reserve/checkpoints differ from fresh S8");
    if (saved.spec.processes.helpers.size() != before.helpers.size() ||
        saved.spec.processes.helpers.size() != after.helpers.size())
        return refusal("spec.processes.helpers", "measured S8 helper count changed");
    for (size_t i = 0; i < after.helpers.size(); ++i) {
        const auto& h = saved.spec.processes.helpers[i];
        const auto& was = before.helpers[i];
        const auto& now_budget = after.helpers[i];
        if (h.role != was.role || h.gpu != was.bdf || h.role != now_budget.role || h.gpu != now_budget.bdf ||
            h.vram_budget_mib > was.expert_budget_mib || h.vram_budget_mib > now_budget.expert_budget_mib)
            return refusal("spec.processes.helpers",
                           "current S8 helper budget cannot admit measured owner map");
    }
    if (bool(before.transport) != bool(saved.spec.handoff) || bool(after.transport) != bool(saved.spec.handoff))
        return refusal("spec.handoff", "measured remote transport presence differs from current S8");
    if (saved.spec.handoff) {
        const auto& t = saved.spec.handoff->transport;
        const auto matches = [&](const v2::S8Resolution& r) {
            const auto& h = *r.transport;
            return h.shm == t.shm && h.max_rows == t.max_rows && h.small_rows == t.small_rows &&
                   h.gemv_large_rows == t.gemv_large_rows && h.skip_owned == t.skip_owned && r.kernels &&
                   r.kernels->moe_down_tile == saved.spec.handoff->kernels.moe_down_tile && r.telemetry &&
                   r.telemetry->period_s == saved.spec.handoff->telemetry.period_s;
        };
        if (!matches(before) || !matches(after))
            return refusal("spec.handoff",
                           "fresh S8 transport, kernels or telemetry differs from stored semantics");
    }
    for (const auto& row : before.calibration_entries) {
        const auto present = std::find_if(saved.spec.calibration.begin(), saved.spec.calibration.end(),
                                          [&](const auto& entry) { return entry.key == row.key; });
        if (present == saved.spec.calibration.end() || !same_calibration(*present, row))
            return refusal("spec.calibration",
                           "writer's stored S8 evidence differs from original P1 resolution");
    }
    if (auto why = check_selected_evidence(before.calibration_entries, after.calibration_entries, records, now))
        return *why;
    return std::nullopt;
}
} // namespace

StoredS8Evidence read_stored_s8_evidence(const std::filesystem::path& path,
                                         const std::string& expected_digest) {
    namespace fs = std::filesystem;
    constexpr uint64_t max_bytes = 16ULL * 1024ULL * 1024ULL;
    if (!fs::is_regular_file(path) || fs::is_symlink(path) || fs::file_size(path) > max_bytes)
        throw std::runtime_error("replay: bounded native S8 producer record is required");
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("replay: cannot read native S8 producer record");
    std::string bytes(static_cast<size_t>(max_bytes + 1), '\0');
    in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    const auto count = in.gcount();
    if (count > static_cast<std::streamsize>(max_bytes) || in.bad())
        throw std::runtime_error("replay: native S8 evidence exceeds bound or could not be read");
    bytes.resize(static_cast<size_t>(count));
    const auto decoded = v2::parse_strict_json(bytes, path.string());
    if (!decoded)
        throw std::runtime_error("replay: invalid native S8 producer record: " + decoded.error().message);
    const Json& doc = decoded.value();
    if (!doc.is_object() || doc.size() != 5 || doc.value("version", 0) != 1 ||
        !doc.contains("manifest_digest") || !doc.at("manifest_digest").is_string() ||
        doc.at("manifest_digest") != expected_digest || !doc.contains("produced_at_unix_s") ||
        !doc.at("produced_at_unix_s").is_number_integer() || !doc.contains("p1_record") ||
        !doc.at("p1_record").is_object() ||
        doc.at("p1_record").value("kind", "") != "bc_serve_v2_measured_inputs" ||
        !doc.contains("calibration_records") || !doc.at("calibration_records").is_array())
        throw std::runtime_error(
            "replay: native P1/S8 evidence is incomplete or not bound to admitted manifest");
    const auto produced = doc.at("produced_at_unix_s").get<int64_t>();
    if (produced <= 0)
        throw std::runtime_error("replay: P1 measurement issuance has an invalid Unix time");
    return {doc.at("p1_record"), doc.at("calibration_records"), expected_digest, produced};
}

v2::Status verify_stored_s8(const v2::Manifest& admitted, const Resolved& current_config, const Json& facts,
                            const v2::P1LiveRoom& live, const StoredS8Evidence& issued,
                            const std::filesystem::path& engine, const std::filesystem::path& helper,
                            int64_t now) {
    if (now <= 0 || now < issued.produced_at_unix_s)
        return refusal("calibration.now_unix_s", "replay's current time precedes its measured evidence");
    if (!admitted.spec.plan.resource_inputs || !admitted.spec.plan.engine_memory)
        return refusal("spec.plan.resource_inputs", "stored manifest lacks measured S8 resources");
    if (issued.p1_record.value("kind", "") != "bc_serve_v2_measured_inputs")
        return refusal("p1_record", "replay needs the real native P1 producer record");
    v2::WriterInputs original;
    original.p1_record = issued.p1_record;
    original.calibration_records = issued.calibration_records;
    original.power_profile = admitted.spec.intent.power.profile;
    original.now_unix_s = issued.produced_at_unix_s;
    auto base = v2::s10_inputs(admitted, current_config, original);
    if (!base)
        return base.error();
    auto baseline = v2::resolve_s8(base.value().measured, base.value().calibrations, original.now_unix_s);
    if (!baseline)
        return baseline.error();
    // Replay can never substitute old measured allocation data for new
    // operator --sets. Only the live bytes/room and actual current S8 scope
    // change; the native producer's old measurement remains the admitted plan.
    v2::WriterInputs fresh = original;
    fresh.now_unix_s = now;
    Json& p1 = fresh.p1_record;
    const auto& old_scope = issued.p1_record.at("scope");
    auto& scope = p1.at("scope");
    scope["model_digest"] = container_identity(admitted.spec.intent.model);
    scope["engine_build_digest"] =
        engine_identity(engine, admitted.spec.processes.engine.runtime.radiance_home);
    scope["helper_build_digest"] =
        admitted.spec.processes.helpers.empty() ? sha256_hex(Json::array().dump()) : file_sha256(helper);
    scope["layout_digest"] = sha256_hex(current_config.values.at("layout").dump());
    scope["inventory_digest"] = sha256_hex(facts.at("gpus").dump());
    scope["owner_map_digest"] = admitted.spec.plan.owner_map.owner_map_sha256;
    scope["context_target_tokens"] = current_config.values.at("service").at("max_context");
    scope["retention_target_tokens"] = admitted.spec.plan.resource_inputs->retention_target_tokens;
    scope["kv_bytes_per_token"] = admitted.spec.plan.resource_inputs->kv_bytes_per_token;
    if (scope != old_scope)
        return refusal(
            "p1.scope",
            "current measured model/build/owner-map/service/inventory scope differs from S8 evidence");
    Json& resources = p1.at("resource_inputs");
    resources["mem_available_bytes"] = live.mem_available_bytes;
    resources["cgroup_headroom_bytes"] = live.cgroup_headroom_bytes;
    resources["pin_available_bytes"] = live.pin_available_bytes;
    for (auto& rank : p1.at("gpu")) {
        const std::string bdf = rank.at("bdf").get<std::string>();
        const auto present = std::find_if(facts.at("gpus").begin(), facts.at("gpus").end(),
                                          [&](const Json& g) { return g.at("bdf") == bdf; });
        if (present == facts.at("gpus").end())
            return refusal("p1.gpu", "current engine GPU vanished from measured S8 scope");
        constexpr int64_t mib = 1024 * 1024;
        const int64_t available = present->at("vram_available_mib").get<int64_t>();
        if (available < 0 || available > std::numeric_limits<int64_t>::max() / mib)
            return refusal("p1.gpu", "current engine live VRAM is invalid");
        rank["claimable_bytes"] = std::min<int64_t>(rank.at("claimable_bytes").get<int64_t>(), available * mib);
    }
    v2::Manifest now_manifest = admitted;
    for (auto& gpu : now_manifest.spec.facts.gpus) {
        const auto present = std::find_if(facts.at("gpus").begin(), facts.at("gpus").end(),
                                          [&](const Json& g) { return g.at("bdf") == gpu.bdf; });
        if (present != facts.at("gpus").end())
            gpu.vram.available_mib = present->at("vram_available_mib").get<int64_t>();
    }
    auto current = v2::s10_inputs(now_manifest, current_config, fresh);
    if (!current)
        return current.error();
    auto requalified = v2::resolve_s8(current.value().measured, current.value().calibrations, now);
    if (!requalified)
        return requalified.error();
    return same_s8(admitted, baseline.value(), requalified.value(), current.value().calibrations, now);
}

} // namespace bc::serving::session
