#include "stored_manifest.h"

#include <algorithm>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace bc::serving::session {
namespace {
using Json = nlohmann::json;
using Code = v2::ConfigErrorCode;
constexpr int64_t MiB = 1024LL * 1024LL;
constexpr int64_t MaxBytes = std::numeric_limits<int64_t>::max() / 2;

ReplayError refuse(ReplayRefusal reason, Code code, std::string path, std::string message) {
    return {reason, v2::make_error(code, std::move(path), std::move(message))};
}

v2::Status no_room(std::string path, std::string description) {
    return v2::make_error(Code::admission_refused, std::move(path), std::move(description));
}

bool valid_bytes(int64_t n) {
    return n >= 0 && n <= MaxBytes;
}
bool add_bytes(int64_t n, int64_t* total) {
    if (!valid_bytes(n) || n > MaxBytes - *total)
        return false;
    *total += n;
    return true;
}
bool add_mib(int64_t n, int64_t* total) {
    return n >= 0 && n <= MaxBytes / MiB && add_bytes(n * MiB, total);
}

const Json* find_gpu(const Json& gpus, const std::string& bdf) {
    if (!gpus.is_array())
        return nullptr;
    const Json* found = nullptr;
    for (const auto& g : gpus) {
        if (!g.is_object() || !g.contains("bdf") || !g.at("bdf").is_string())
            return nullptr;
        if (g.at("bdf") == bdf) {
            if (found)
                return nullptr;
            found = &g;
        }
    }
    return found;
}

std::vector<std::string> selected(const v2::Manifest& m) {
    auto names = m.spec.processes.engine.device.gpus;
    for (const auto& helper : m.spec.processes.helpers)
        names.push_back(helper.gpu);
    return names;
}
} // namespace

Json execution_projection(const Json& spec) {
    Json projected = {{"intent", spec.at("intent")},
                      {"processes", spec.at("processes")},
                      {"power", spec.at("plan").at("power")},
                      {"host_reserve_mib", spec.at("resources").at("host").at("reserve_mib")}};
    if (spec.contains("handoff")) {
        Json handoff = spec.at("handoff");
        // The shared segment is an execution-record identity, not a
        // configuration setting. The current owner still checks its namespace.
        handoff.at("transport").erase("shm");
        // This is an S10 startup decision, checked by live calibration and
        // runtime policy gates; it is not a static config setting.
        handoff.at("placement").erase("live_policy_state");
        projected["handoff"] = std::move(handoff);
    }
    return projected;
}

ReplayResult authenticate_replay(std::string_view bytes, std::string_view source,
                                 const ReplayEvidence& evidence, CalibrationVerification calibration) {
    const auto decoded = v2::read_manifest(bytes, std::string(source), v2::PlanUse::serving);
    if (!decoded)
        return ReplayError{ReplayRefusal::bytes_or_digest, decoded.error()};
    const auto& m = decoded.value();
    if (evidence.producer_sha256.empty() || evidence.model_identity.empty() ||
        evidence.engine_identity.empty() || !evidence.fresh_facts.is_object() ||
        !evidence.fresh_facts.contains("gpus") || !evidence.effective_spec.is_object())
        return refuse(
            ReplayRefusal::missing_gate, Code::invalid_state, "replay.evidence",
            "build, model, fresh device inventory and effective RSC12 projection must all be checked");

    if (m.metadata.producer.build_sha256 != evidence.producer_sha256)
        return refuse(ReplayRefusal::build, Code::abi_mismatch, "metadata.producer.build_sha256",
                      "stored manifest was produced by a different bc-serve build");
    if (!m.spec.plan.engine_memory || m.spec.plan.engine_memory->model_identity != evidence.model_identity)
        return refuse(ReplayRefusal::model, Code::plan_mismatch, "spec.plan.engine_memory.model_identity",
                      "model artifact identity changed");
    if (m.spec.plan.engine_memory->engine_identity != evidence.engine_identity)
        return refuse(ReplayRefusal::build, Code::abi_mismatch, "spec.plan.engine_memory.engine_identity",
                      "engine, kernels or helper build changed");

    try {
        const Json source_doc = Json::parse(bytes);
        const Json& saved_spec = source_doc.at("spec");
        const auto& recorded_cards = saved_spec.at("facts").at("gpus");
        const auto& now_cards = evidence.fresh_facts.at("gpus");
        std::set<std::string> distinct;
        for (const auto& bdf : selected(m)) {
            if (!distinct.insert(bdf).second)
                return refuse(ReplayRefusal::devices, Code::ownership_mismatch, "spec.processes",
                              "an engine/helper BDF appears twice");
            const auto* then = find_gpu(recorded_cards, bdf);
            const auto* now = find_gpu(now_cards, bdf);
            if (!then || !now || then->at("arch") != now->at("arch") ||
                then->at("hip_ordinal") != now->at("hip_ordinal") ||
                then->at("vram").at("total_mib") != now->at("vram_total_mib"))
                return refuse(ReplayRefusal::devices, Code::ownership_mismatch, "spec.facts.gpus",
                              "selected card BDF, architecture, HIP ordinal or total VRAM changed: " + bdf);
        }
        if (calibration == CalibrationVerification::exact_draft_match &&
            saved_spec.at("calibration") != evidence.effective_spec.at("calibration"))
            return refuse(ReplayRefusal::calibration, Code::calibration_invalid, "spec.calibration",
                          "stored calibration no longer matches current seeded/refit records");
        if (execution_projection(saved_spec) != execution_projection(evidence.effective_spec))
            return refuse(ReplayRefusal::execution_projection, Code::plan_mismatch, "spec",
                          "stored semantic execution projection differs from the current config/variants");
    } catch (const std::exception& error) {
        return refuse(ReplayRefusal::missing_gate, Code::invalid_state, "replay.evidence",
                      std::string("cannot validate current projection or inventory: ") + error.what());
    }
    return m;
}

v2::Status check_replay_live_room(const v2::Manifest& m, const Json& fresh, const v2::P1LiveRoom& room) {
    if (!fresh.is_object() || !fresh.contains("gpus") || !fresh.contains("host") ||
        !fresh.at("host").contains("memlock"))
        return no_room("replay.live", "current GPU and host/memlock facts are mandatory");
    const auto& host = m.spec.resources.host;
    const auto& inputs = m.spec.plan.resource_inputs;
    if (!inputs || !m.spec.plan.engine_memory)
        return no_room("replay.live.plan", "stored manifest has no measured admission inputs");
    if (room.mem_available_bytes <= 0 || room.cgroup_headroom_bytes <= 0 || room.pin_available_bytes <= 0)
        return no_room("replay.live.host", "fresh host, cgroup or pin measurement absent");

    int64_t allocated = 0, pinned = 0;
    if (!add_mib(host.pool_mib, &allocated) || !add_mib(host.reserve_mib, &allocated) ||
        !add_mib(host.pool_mib, &pinned) || !add_mib(host.kv_mib, &pinned))
        return no_room("replay.live.host", "stored allocation overflows checked byte arithmetic");
    // The measured P1 producer is the authority for the host allocations
    // still to come, not a guessed constant or stale MemAvailable.
    for (int64_t n : {inputs->host_embedding_bytes, inputs->unowned_expert_bytes, inputs->staging_bytes,
                      inputs->ngram_bytes, inputs->checkpoint_bytes, inputs->resident_other_bytes})
        if (!add_bytes(n, &allocated))
            return no_room("replay.live.host", "measured host allocations overflow");
    if (allocated > room.mem_available_bytes || allocated > room.cgroup_headroom_bytes)
        return no_room("replay.live.host", "live MemAvailable/cgroup room cannot hold the stored plan");
    if (pinned > room.pin_available_bytes)
        return no_room("replay.live.pin",
                       "live pinned-memory headroom cannot hold the stored pool and KV tier");

    const Json& memlock = fresh.at("host").at("memlock");
    if (!memlock.is_object() || !memlock.contains("limit_mib"))
        return no_room("replay.live.memlock", "RLIMIT_MEMLOCK was not detected");
    if (!memlock.at("limit_mib").is_null()) {
        const auto& lim = memlock.at("limit_mib");
        if (!lim.is_number_integer() || lim.get<int64_t>() < 0 || lim.get<int64_t>() > MaxBytes / MiB ||
            pinned > lim.get<int64_t>() * MiB)
            return no_room("replay.live.memlock", "live memlock limit does not admit the stored pinned demand");
    }

    const auto& cards = m.spec.plan.engine_memory->expected.vram.cards;
    const auto& engine = m.spec.processes.engine.device.gpus;
    if (cards.size() != engine.size())
        return no_room("replay.live.vram", "rank memory and selected card counts differ");
    for (size_t i = 0; i < engine.size(); ++i) {
        const Json* now = find_gpu(fresh.at("gpus"), engine[i]);
        if (!now || !now->contains("vram_available_mib") || !now->at("vram_available_mib").is_number_integer())
            return no_room("replay.live.vram", "missing live rank VRAM: " + engine[i]);
        const int64_t available = now->at("vram_available_mib").get<int64_t>();
        if (available < 0 || available > MaxBytes / MiB || cards[i].free < 0 || available * MiB < cards[i].free)
            return no_room("replay.live.vram",
                           "rank " + std::to_string(i) + " has less claimable VRAM than measured");
    }
    if (!m.spec.plan.resource_inputs ||
        m.spec.plan.resource_inputs->helper_footprints.size() != m.spec.processes.helpers.size())
        return no_room("replay.live.helpers", "measured per-helper worker footprints are required");
    for (size_t i = 0; i < m.spec.processes.helpers.size(); ++i) {
        const auto& h = m.spec.processes.helpers[i];
        const Json* now = find_gpu(fresh.at("gpus"), h.gpu);
        if (!now || !now->contains("vram_available_mib") || !now->at("vram_available_mib").is_number_integer())
            return no_room("replay.live.helpers", "missing helper VRAM: " + h.gpu);
        const int64_t available = now->at("vram_available_mib").get<int64_t>();
        int64_t required = 0;
        const auto& footprint = inputs->helper_footprints[i];
        if (footprint.role != h.role || footprint.bdf != h.gpu || !add_mib(h.vram_budget_mib, &required) ||
            !add_bytes(footprint.total_bytes, &required) || available < 0 || available > MaxBytes / MiB ||
            available * MiB < required)
            return no_room("replay.live.helpers", "helper " + h.role + " has insufficient claimable VRAM");
    }
    return std::nullopt;
}

} // namespace bc::serving::session
