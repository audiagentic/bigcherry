// RSC02 S5: card-free strict-v2 replay authentication and per-start admission.
// The fixture is the committed v2 manifest; no device, chmod or lease is touched.
#include "stored_manifest.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <variant>

#ifndef BC_CONTRACT_V2_DOCS_DIR
#error "BC_CONTRACT_V2_DOCS_DIR must name the committed strict-v2 manifest fixtures"
#endif

namespace fs = std::filesystem;
namespace ss = bc::serving::session;
namespace v2 = bc::serving::v2;
using Json = nlohmann::json;

namespace {
std::string load(const fs::path& name) {
    std::ifstream in(name, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot read fixture");
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
void require(bool ok, const std::string& why) {
    if (!ok)
        throw std::runtime_error(why);
}
void refused(const ss::ReplayResult& result, ss::ReplayRefusal kind, v2::ConfigErrorCode code) {
    const auto* e = std::get_if<ss::ReplayError>(&result);
    require(e && e->reason == kind && e->detail.code == code, "unexpected replay result/typed refusal");
}
ss::ReplayEvidence fresh(const Json& stored) {
    ss::ReplayEvidence e;
    e.producer_sha256 = stored.at("metadata").at("producer").at("build_sha256");
    const auto& plan = stored.at("spec").at("plan").at("engine_memory");
    e.model_identity = plan.at("model_identity");
    e.engine_identity = plan.at("engine_identity");
    e.effective_spec = stored.at("spec");
    Json gpus = Json::array();
    for (const auto& g : stored.at("spec").at("facts").at("gpus"))
        gpus.push_back({{"bdf", g.at("bdf")},
                        {"arch", g.at("arch")},
                        {"hip_ordinal", g.at("hip_ordinal")},
                        {"vram_total_mib", g.at("vram").at("total_mib")},
                        {"vram_available_mib", g.at("vram").at("total_mib")}});
    e.fresh_facts = {{"gpus", gpus}, {"host", {{"memlock", {{"limit_mib", nullptr}}}}}};
    return e;
}
} // namespace

int main() {
    try {
        const fs::path path = fs::path(BC_CONTRACT_V2_DOCS_DIR) / "example.flashnext-4gpu.json";
        const std::string bytes = load(path);
        const Json source = Json::parse(bytes);
        const auto evidence = fresh(source);
        const auto accepted = ss::authenticate_replay(bytes, path.string(), evidence);
        require(std::holds_alternative<v2::Manifest>(accepted),
                "valid unchanged strict-v2 manifest must authenticate");
        require(load(path) == bytes, "replay must not rewrite even a single byte");

        Json tampered = source;
        tampered["spec"]["intent"]["model"] = "changed-without-rehashing";
        refused(ss::authenticate_replay(tampered.dump(), "tampered.json", evidence),
                ss::ReplayRefusal::bytes_or_digest, v2::ConfigErrorCode::digest_mismatch);
        tampered = source;
        tampered["metadata"]["content_digest"] = "sha256:" + std::string(64, '0');
        refused(ss::authenticate_replay(tampered.dump(), "wrong-digest.json", evidence),
                ss::ReplayRefusal::bytes_or_digest, v2::ConfigErrorCode::digest_mismatch);

        auto e = evidence;
        e.producer_sha256 = std::string(64, 'a');
        refused(ss::authenticate_replay(bytes, path.string(), e), ss::ReplayRefusal::build,
                v2::ConfigErrorCode::abi_mismatch);
        e = evidence;
        e.model_identity = std::string(64, 'b');
        refused(ss::authenticate_replay(bytes, path.string(), e), ss::ReplayRefusal::model,
                v2::ConfigErrorCode::plan_mismatch);
        e = evidence;
        e.engine_identity = std::string(64, 'c');
        refused(ss::authenticate_replay(bytes, path.string(), e), ss::ReplayRefusal::build,
                v2::ConfigErrorCode::abi_mismatch);
        e = evidence;
        e.fresh_facts["gpus"][0]["arch"] = "gfx-foreign";
        refused(ss::authenticate_replay(bytes, path.string(), e), ss::ReplayRefusal::devices,
                v2::ConfigErrorCode::ownership_mismatch);
        e = evidence;
        e.fresh_facts["gpus"][0]["hip_ordinal"] = 999;
        refused(ss::authenticate_replay(bytes, path.string(), e), ss::ReplayRefusal::devices,
                v2::ConfigErrorCode::ownership_mismatch);
        e = evidence;
        e.effective_spec["calibration"][0]["evidence"] = "stale refit";
        refused(ss::authenticate_replay(bytes, path.string(), e), ss::ReplayRefusal::calibration,
                v2::ConfigErrorCode::calibration_invalid);
        e = evidence;
        e.effective_spec["processes"]["engine"]["server"]["port"] = 9999;
        refused(ss::authenticate_replay(bytes, path.string(), e), ss::ReplayRefusal::execution_projection,
                v2::ConfigErrorCode::plan_mismatch);
        // A production start with the live policy on stores the policy the start derived (from the calibration store, which
        // the replay key already binds, and the measured plan). The replay's current draft is unmeasured and derives none:
        // that absence is not a configuration change and must not refuse a stored policy-on manifest.
        e = evidence;
        e.effective_spec["handoff"]["placement"].erase("live_policy");
        require(std::holds_alternative<v2::Manifest>(ss::authenticate_replay(bytes, path.string(), e)),
                "a stored production manifest with a derived live policy must replay against a draft that derives none");
        // Real configuration changes still refuse, with the policy present or not.
        e.effective_spec["handoff"]["placement"]["strategy"] = "roundrobin";
        refused(ss::authenticate_replay(bytes, path.string(), e), ss::ReplayRefusal::execution_projection,
                v2::ConfigErrorCode::plan_mismatch);
        e = evidence;
        e.effective_spec["handoff"]["placement"]["live_policy"]["cadence_ns"] = 1;
        require(std::holds_alternative<v2::Manifest>(ss::authenticate_replay(bytes, path.string(), e)),
                "a different derived policy value is a startup derivation, not configuration (production)");
        e = evidence;
        e.effective_spec["intent"]["layout"] = "another-layout";
        refused(ss::authenticate_replay(bytes, path.string(), e), ss::ReplayRefusal::execution_projection,
                v2::ConfigErrorCode::plan_mismatch);
        // A lab manifest's policy is explicit operator input (--live-policy), so it IS part of what must match.
        Json lab_stored = source.at("spec"), lab_draft = source.at("spec");
        lab_stored["intent"]["service"]["profile"] = "lab";
        lab_draft["intent"]["service"]["profile"] = "lab";
        lab_draft["handoff"]["placement"].erase("live_policy");
        require(ss::execution_projection(lab_stored) != ss::execution_projection(lab_draft),
                "a lab manifest's explicit live policy stays in the execution projection");
        refused(ss::authenticate_replay(bytes, path.string(), {}), ss::ReplayRefusal::missing_gate,
                v2::ConfigErrorCode::invalid_state);

        const auto& manifest = std::get<v2::Manifest>(accepted);
        v2::P1LiveRoom room{};
        room.mem_available_bytes = 281474976710656LL;
        room.cgroup_headroom_bytes = 281474976710656LL;
        room.pin_available_bytes = 281474976710656LL;
        require(!ss::check_replay_live_room(manifest, evidence.fresh_facts, room),
                "fresh ample room must admit the stored plan");
        room.mem_available_bytes = 1;
        require(ss::check_replay_live_room(manifest, evidence.fresh_facts, room).has_value(),
                "stored manifest cannot skip current MemAvailable");
        room.mem_available_bytes = 281474976710656LL;
        room.cgroup_headroom_bytes = 1;
        require(ss::check_replay_live_room(manifest, evidence.fresh_facts, room).has_value(),
                "stored manifest cannot skip cgroup");
        room.cgroup_headroom_bytes = 281474976710656LL;
        room.pin_available_bytes = 1;
        require(ss::check_replay_live_room(manifest, evidence.fresh_facts, room).has_value(),
                "stored manifest cannot skip pin room");
        room.pin_available_bytes = 281474976710656LL;
        e = evidence;
        e.fresh_facts["host"]["memlock"]["limit_mib"] = 0;
        require(ss::check_replay_live_room(manifest, e.fresh_facts, room).has_value(),
                "stored manifest cannot skip live memlock");
        e = evidence;
        e.fresh_facts["gpus"][0]["vram_available_mib"] = 0;
        require(ss::check_replay_live_room(manifest, e.fresh_facts, room).has_value(),
                "stored manifest cannot skip per-rank live VRAM");
        std::cout << "stored replay: all typed refusals, original bytes and live gates passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RSC02 S5 stored-manifest test failed: " << error.what() << '\n';
        return 1;
    }
}
