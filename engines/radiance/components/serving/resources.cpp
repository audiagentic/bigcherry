#include "resources.h"

#include "calibration.h"
#include "experts.h"
#include "serving.h"
#include "v2/manifest.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bc::serving::resources {
namespace {

using Json = nlohmann::json;
using v2::ConfigErrorCode;
constexpr uint64_t MiB = uint64_t{1024} * 1024;

v2::ConfigError invalid(std::string message) {
    return v2::make_error(ConfigErrorCode::invalid_value, "resources", std::move(message));
}
v2::ConfigError refused(std::string message) {
    return v2::make_error(ConfigErrorCode::admission_refused, "resources", std::move(message));
}
v2::ConfigError malformed(std::string message) {
    return v2::make_error(ConfigErrorCode::invalid_format, "resources", std::move(message));
}
std::optional<uint64_t> add(uint64_t lhs, uint64_t rhs) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
        return std::nullopt;
    return lhs + rhs;
}
std::optional<uint64_t> multiply(uint64_t lhs, uint64_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
        return std::nullopt;
    return lhs * rhs;
}
uint64_t ceil_div(uint64_t numerator, uint64_t denominator) {
    return (numerator / denominator) + (numerator % denominator != 0 ? 1U : 0U);
}
uint64_t mib(uint64_t value) {
    return ceil_div(value, MiB);
}
bool sha256_hex(std::string_view hash) {
    return hash.size() == 64 && std::all_of(hash.begin(), hash.end(), [](char value) {
               return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
           });
}
uint64_t nonnegative_bytes(const Json& j, std::string_view field) {
    const auto key = std::string(field);
    const Json& value = j.at(key);
    if (!value.is_number_integer() ||
        (value.is_number_integer() && !value.is_number_unsigned() && value.get<int64_t>() < 0)) {
        throw std::invalid_argument(key + " must be a measured, non-negative integer");
    }
    return value.get<uint64_t>();
}
uint64_t positive_bytes(const Json& j, std::string_view field) {
    const uint64_t value = nonnegative_bytes(j, field);
    if (value == 0)
        throw std::invalid_argument(std::string(field) + " must be positive");
    return value;
}
uint64_t calibration_mib(const Calibration& calibration, std::string_view field, const std::string& layout) {
    const auto entry = calibration.find("calibration." + std::string(field), {"", layout, ""});
    // No calibrated target means no retained KV tier for this layout (RSC03 contract).
    if (!entry)
        return 0;
    if (!entry->value.is_number_integer() ||
        (!entry->value.is_number_unsigned() && entry->value.get<int64_t>() < 0)) {
        throw std::invalid_argument("bad calibrated " + std::string(field));
    }
    return entry->value.get<uint64_t>();
}

struct ExpertId {
    int32_t layer = 0;
    int32_t expert = 0;
    auto operator<=>(const ExpertId&) const = default;
};

v2::Result<ModelFootprint> inspect_weights(const std::filesystem::path& model, const Json& owner_map,
                                           bool ngram_in_ram) {
    bc::experts::Container container;
    if (!container.open(model.c_str())) {
        return malformed("cannot open verified RAD expert directory: " + model.string());
    }
    if (container.hdr.str_bytes == 0 || container.hdr.str_bytes > (uint64_t{1} << 28U) ||
        !container.range(container.hdr.str_off, container.hdr.str_bytes)) {
        return malformed("model has invalid RAD name strings");
    }
    std::vector<char> names(static_cast<size_t>(container.hdr.str_bytes));
    if (!container.pread_full(container.hdr.str_off, names.data(), names.size())) {
        return malformed("cannot read RAD weight names");
    }
    std::map<ExpertId, uint64_t> experts;
    ModelFootprint weights;
    weights.owner_map_digest = owner_map.at("owner_map_sha256").get<std::string>();
    for (const auto& entry : container.dir_store) {
        if (entry.name >= names.size() || entry.bytes > container.bytes) {
            return malformed("model has invalid weight directory entries");
        }
        const auto first = names.cbegin() + static_cast<std::ptrdiff_t>(entry.name);
        const auto end = std::find(first, names.cend(), '\0');
        if (end == names.cend())
            return malformed("model contains an unterminated weight name");
        std::string name(first, end);
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
        const uint64_t rounded = ceil_div(entry.bytes, 256);
        const auto aligned = multiply(rounded, 256);
        if (!aligned)
            return invalid("resource byte count overflow");
        if (entry.layer >= 0 && entry.expert >= 0 && entry.slot >= 0) {
            auto& bytes = experts[{entry.layer, entry.expert}];
            const auto updated = add(bytes, *aligned);
            if (!updated)
                return invalid("resource byte count overflow");
            bytes = *updated;
        } else if (name.find("ngram") != std::string::npos) {
            const auto updated = add(weights.ngram_bytes, *aligned);
            if (!updated)
                return invalid("resource byte count overflow");
            weights.ngram_bytes = *updated;
        } else if (name.find("embed") != std::string::npos) {
            const auto updated = add(weights.embedding_bytes, *aligned);
            if (!updated)
                return invalid("resource byte count overflow");
            weights.embedding_bytes = *updated;
        }
    }
    if (experts.empty())
        return malformed("model has no routed expert directory entries");
    std::set<ExpertId> assigned;
    for (const Json& helper : owner_map.at("helpers")) {
        for (const Json& pair : helper.at("experts")) {
            if (!pair.is_array() || pair.size() != 2)
                return malformed("owner map has invalid expert pair");
            const ExpertId id{pair.at(0).get<int32_t>(), pair.at(1).get<int32_t>()};
            if (!experts.contains(id) || !assigned.insert(id).second) {
                return invalid("owner map does not match model expert directory");
            }
        }
    }
    for (const auto& [id, bytes] : experts) {
        if (!assigned.contains(id)) {
            const auto updated = add(weights.unowned_expert_bytes, bytes);
            if (!updated)
                return invalid("resource byte count overflow");
            weights.unowned_expert_bytes = *updated;
        }
        weights.staging_bytes = std::max(weights.staging_bytes, bytes);
    }
    if (!ngram_in_ram)
        weights.ngram_bytes = 0;
    return weights;
}

constexpr std::array<std::pair<std::string_view, std::string_view>, AllocationCount> PartLabels{
    {{"embedding", "engine plan"},
     {"unowned_experts", "published owner map"},
     {"staging", "engine plan"},
     {"pool_margin", "calibrated margin"},
     {"ngram_ram", "engine plan"},
     {"checkpoints", "engine plan"},
     {"other_engine", "engine plan"},
     {"retained_kv", "target * bytes per token"},
     {"host_reserve", "host reserve policy"}}};
} // namespace

v2::Result<ResourceLedger> calculate(const ResourceRequest& request) {
    const auto& model = request.model;
    const auto& target = request.target;
    const auto& host = request.host;
    if (!sha256_hex(model.owner_map_digest)) {
        return invalid("owner_map_digest must identify the actual ownership plan (sha256)");
    }
    if (model.kv_bytes_per_token == 0 || target.context_tokens == 0 || target.concurrent_sessions == 0 ||
        host.mem_available_bytes == 0 || host.cgroup_headroom_bytes == 0 || host.pin_available_bytes == 0 ||
        request.engine_ranks == 0) {
        return invalid("resource measurements and cardinalities must be positive");
    }
    const auto total_tokens = multiply(target.context_tokens, target.concurrent_sessions);
    if (!total_tokens)
        return invalid("resource byte count overflow");
    if (target.retained_tokens > *total_tokens) {
        return invalid("retained KV target exceeds concurrent_sessions * context_tokens");
    }
    ResourceLedger result;
    result.owner_map_digest = model.owner_map_digest;
    std::set<std::string> seen;
    for (const auto& helper : request.helpers) {
        if (!seen.insert(helper.bdf).second)
            return invalid("duplicate helper BDF " + helper.bdf);
        if (helper.vram_mib == 0 || helper.worker_footprint_bytes == 0) {
            return invalid("worker_footprint_bytes and vram_mib must be positive");
        }
        const auto card = multiply(helper.vram_mib, MiB);
        const auto reserve = multiply(helper.architecture_reserve_mib, MiB);
        if (!card || !reserve)
            return invalid("resource byte count overflow");
        if (*reserve > *card || helper.worker_footprint_bytes > *card - *reserve ||
            *card - *reserve - helper.worker_footprint_bytes < MiB) {
            return refused(helper.bdf + ": no expert VRAM after architecture reserve and worker footprint");
        }
        const uint64_t available = *card - *reserve - helper.worker_footprint_bytes;
        const uint64_t budget = available / MiB;
        if (helper.planned_budget_mib && *helper.planned_budget_mib != budget) {
            return invalid(helper.bdf + ": helper budget disagrees with BDF/worker footprint");
        }
        result.helpers.push_back({helper.bdf, helper.arch, helper.vram_mib, helper.architecture_reserve_mib,
                                  mib(helper.worker_footprint_bytes), budget, available - (budget * MiB)});
    }
    const auto first_pool = add(model.embedding_bytes, model.unowned_expert_bytes);
    const auto second_pool = add(model.staging_bytes, target.pool_margin_bytes);
    if (!first_pool || !second_pool)
        return invalid("resource byte count overflow");
    const auto pool_bytes = add(*first_pool, *second_pool);
    const auto kv_bytes = multiply(target.retained_tokens, model.kv_bytes_per_token);
    if (!pool_bytes || !kv_bytes)
        return invalid("resource byte count overflow");
    result.host_pool_mib = mib(*pool_bytes);
    result.host_kv_mib = mib(*kv_bytes);
    const auto pool_total_mib = multiply(result.host_pool_mib, request.engine_ranks);
    const auto pinned_mib = pool_total_mib ? add(*pool_total_mib, result.host_kv_mib) : std::nullopt;
    if (!pinned_mib)
        return invalid("resource byte count overflow");
    const auto pinned_bytes = multiply(*pinned_mib, MiB);
    const auto unmanaged = add(model.ngram_bytes, model.checkpoint_bytes);
    const auto engine = add(unmanaged.value_or(0), model.other_engine_bytes);
    if (!pinned_bytes || !unmanaged || !engine)
        return invalid("resource byte count overflow");
    const auto with_engine = add(*pinned_bytes, *engine);
    if (!with_engine)
        return invalid("resource byte count overflow");
    const auto committed = add(*with_engine, target.host_reserve_bytes);
    if (!committed)
        return invalid("resource byte count overflow");
    result.pin_committed_bytes = *pinned_bytes;
    result.committed_bytes = *committed;
    result.available_bytes = std::min(host.mem_available_bytes, host.cgroup_headroom_bytes);
    result.pin_available_bytes = host.pin_available_bytes;
    if (result.committed_bytes > result.available_bytes) {
        return refused("host RAM admission refused: " + std::to_string(result.committed_bytes) +
                       " bytes required, " + std::to_string(result.available_bytes) +
                       " bytes available (including reserve)");
    }
    if (result.pin_committed_bytes > result.pin_available_bytes) {
        return refused("pin admission refused: " + std::to_string(result.pin_committed_bytes) +
                       " bytes required, " + std::to_string(result.pin_available_bytes) + " bytes available");
    }
    result.remaining_bytes = result.available_bytes - result.committed_bytes;
    const auto embedding_bytes = multiply(model.embedding_bytes, request.engine_ranks);
    const auto unowned_bytes = multiply(model.unowned_expert_bytes, request.engine_ranks);
    const auto staging_bytes = multiply(model.staging_bytes, request.engine_ranks);
    const auto margin_bytes = multiply(target.pool_margin_bytes, request.engine_ranks);
    if (!embedding_bytes || !unowned_bytes || !staging_bytes || !margin_bytes)
        return invalid("per-rank host-pool component overflow");
    result.allocations = {*embedding_bytes,        *unowned_bytes,
                          *staging_bytes,         *margin_bytes,
                          model.ngram_bytes,        model.checkpoint_bytes,
                          model.other_engine_bytes, *kv_bytes,
                          target.host_reserve_bytes};
    return result;
}

v2::Result<ResourceLedger> derive_launch(const LaunchDocuments& documents,
                                         const std::filesystem::path& calibration_cache) {
    const Json& values = documents.resolved_values;
    const Json& resolved_helpers = documents.resolved_helpers;
    const Json& facts = documents.detected_facts;
    const Json& manifest = documents.unmeasured_manifest;
    try {
        const Json& owner = manifest.at("spec").at("plan").at("owner_map");
        if (owner.at("helpers").size() != resolved_helpers.size()) {
            return invalid("owner-map helper count differs from resolved BDF order");
        }
        const std::string layout = values.at("layout").at("name").get<std::string>();
        const Calibration calibration = Calibration::standard(calibration_cache);
        const uint64_t kv_target = calibration_mib(calibration, "host_kv_tier.target_mib", layout);
        const uint64_t margin = calibration_mib(calibration, "host_ledger_margin.mib", layout);
        ResourceRequest request;
        request.engine_ranks = positive_bytes(values.at("engine"), "tp");
        request.target.context_tokens = positive_bytes(values.at("service"), "max_context");
        request.target.concurrent_sessions = positive_bytes(values.at("service"), "max_sessions");
        const auto total = multiply(request.target.context_tokens, request.target.concurrent_sessions);
        const auto kv_target_bytes = multiply(kv_target, MiB);
        const auto margin_bytes = multiply(margin, MiB);
        if (!total || !kv_target_bytes || !margin_bytes)
            return invalid("resource byte count overflow");
        const uint64_t kv_per = std::max<uint64_t>(1, ceil_div(*kv_target_bytes, *total));
        request.target.retained_tokens = std::min(*total, *kv_target_bytes / kv_per);
        request.target.pool_margin_bytes = *margin_bytes;
        const auto reserve = multiply(nonnegative_bytes(values.at("memory"), "host_reserve_mib"), MiB);
        if (!reserve)
            return invalid("resource byte count overflow");
        request.target.host_reserve_bytes = *reserve;
        const bool ngram_in_ram = values.at("engine").at("ngram_placement") == "ram";
        const auto model =
            inspect_weights(values.at("engine").at("model").get<std::string>(), owner, ngram_in_ram);
        if (!model)
            return model.error();
        request.model = model.value();
        request.model.kv_bytes_per_token = kv_per;

        for (const Json& helper : resolved_helpers) {
            const std::string bdf = helper.at("bdf").get<std::string>();
            const Json* device = nullptr;
            for (const Json& gpu : facts.at("gpus")) {
                if (gpu.at("bdf") == bdf)
                    device = &gpu;
            }
            if (device == nullptr)
                return invalid("helper BDF absent from current inventory: " + bdf);
            if (helper.at("arch") != device->at("arch") ||
                helper.at("hip_ordinal") != device->at("hip_ordinal")) {
                return invalid("helper role changed since BDF inventory detection: " + bdf);
            }
            HelperCapacity capacity;
            capacity.bdf = bdf;
            capacity.arch = helper.at("arch").get<std::string>();
            capacity.vram_mib = positive_bytes(*device, "vram_total_mib");
            capacity.architecture_reserve_mib = arch_reserve_mib(capacity.arch);
            const uint64_t budget = nonnegative_bytes(helper, "budget_mib");
            if (budget >= capacity.vram_mib ||
                capacity.architecture_reserve_mib >= capacity.vram_mib - budget) {
                return invalid("helper has no model-derived worker footprint: " + bdf);
            }
            const auto footprint =
                multiply(capacity.vram_mib - capacity.architecture_reserve_mib - budget, MiB);
            if (!footprint)
                return invalid("resource byte count overflow");
            capacity.worker_footprint_bytes = *footprint;
            capacity.planned_budget_mib = budget;
            request.helpers.push_back(std::move(capacity));
        }
        const Json& host = facts.at("host");
        const uint64_t memory_mib = positive_bytes(host, "mem_available_mib");
        uint64_t cgroup_mib = memory_mib;
        for (const Json& group : host.value("memory_cgroups", Json::array())) {
            cgroup_mib = std::min(cgroup_mib, positive_bytes(group, "room_mib"));
        }
        const Json& pin = host.at("amdgpu_pin");
        const uint64_t pin_mib = pin.value("loaded", false) && !pin.value("lifted", false)
                                     ? positive_bytes(pin, "cap_mib")
                                     : positive_bytes(host, "mem_total_mib");
        const auto available = multiply(memory_mib, MiB);
        const auto cgroup = multiply(cgroup_mib, MiB);
        const auto pin_bytes = multiply(pin_mib, MiB);
        if (!available || !cgroup || !pin_bytes)
            return invalid("resource byte count overflow");
        request.host = {*available, *cgroup, *pin_bytes};
        return calculate(request);
    } catch (const std::exception& error) {
        // Only external JSON and calibration decoding throws; admission is a typed result.
        return malformed(std::string("invalid resource inputs: ") + error.what());
    }
}

namespace {
v2::Status preflight_host_pool_mib(uint64_t per_rank_mib, uint64_t ranks, uint64_t kv_mib,
                                   uint64_t reserve_mib, const Json& facts) {
    try {
        if (ranks == 0)
            return invalid("engine rank count must be positive");
        const auto pool_total_mib = multiply(per_rank_mib, ranks);
        const auto pinned_mib = pool_total_mib ? add(*pool_total_mib, kv_mib) : std::nullopt;
        const auto committed_mib = pinned_mib ? add(*pinned_mib, reserve_mib) : std::nullopt;
        const auto pinned_bytes = pinned_mib ? multiply(*pinned_mib, MiB) : std::nullopt;
        const auto committed_bytes = committed_mib ? multiply(*committed_mib, MiB) : std::nullopt;
        if (!pool_total_mib || !pinned_mib || !committed_mib || !pinned_bytes || !committed_bytes)
            return invalid("per-rank host-pool preflight overflows measured byte range");

        const Json& host = facts.at("host");
        uint64_t available_mib = positive_bytes(host, "mem_available_mib");
        for (const Json& group : host.value("memory_cgroups", Json::array()))
            available_mib = std::min(available_mib, positive_bytes(group, "room_mib"));
        const uint64_t pin_mib = host.at("amdgpu_pin").value("loaded", false) &&
                                         !host.at("amdgpu_pin").value("lifted", false)
                                     ? positive_bytes(host.at("amdgpu_pin"), "cap_mib")
                                     : positive_bytes(host, "mem_total_mib");
        const auto available_bytes = multiply(available_mib, MiB);
        const auto pin_available_bytes = multiply(pin_mib, MiB);
        if (!available_bytes || !pin_available_bytes)
            return invalid("host-pool preflight room overflows measured byte range");
        if (*committed_bytes > *available_bytes)
            return refused("host RAM admission refused for host pool across " + std::to_string(ranks) +
                           " engine ranks: " + std::to_string(*committed_bytes) + " bytes required, " +
                           std::to_string(*available_bytes) + " bytes available including reserve");
        if (*pinned_bytes > *pin_available_bytes)
            return refused("pin admission refused for host pool across " + std::to_string(ranks) +
                           " engine ranks: " + std::to_string(*pinned_bytes) + " bytes required, " +
                           std::to_string(*pin_available_bytes) + " bytes available");
        return std::nullopt;
    } catch (const std::exception& error) {
        return malformed(std::string("invalid host-pool preflight inputs: ") + error.what());
    }
}
} // namespace

v2::Status preflight_host_pool(const v2::Manifest& manifest, const Json& facts) {
    const auto& host = manifest.spec.resources.host;
    const int64_t ranks = manifest.spec.processes.engine.scheduler.tp;
    if (host.pool_mib < 0 || host.kv_mib < 0 || host.reserve_mib < 0 || ranks <= 0)
        return malformed("invalid host-pool preflight values in admitted measuring manifest");
    return preflight_host_pool_mib(static_cast<uint64_t>(host.pool_mib), static_cast<uint64_t>(ranks),
                                   static_cast<uint64_t>(host.kv_mib),
                                   static_cast<uint64_t>(host.reserve_mib), facts);
}

Json to_json(const ResourceLedger& ledger) {
    Json helpers = Json::array();
    for (const auto& helper : ledger.helpers) {
        helpers.push_back({{"bdf", helper.bdf},
                           {"arch", helper.arch},
                           {"vram_mib", helper.vram_mib},
                           {"reserve_mib", helper.reserve_mib},
                           {"worker_footprint_mib", helper.worker_footprint_mib},
                           {"expert_budget_mib", helper.expert_budget_mib},
                           {"slack_bytes", helper.slack_bytes}});
    }
    Json components = Json::array();
    for (size_t index = 0; index < PartLabels.size(); ++index) {
        components.push_back({{"name", PartLabels[index].first},
                              {"bytes", ledger.allocations[index]},
                              {"source", PartLabels[index].second}});
    }
    return {{"owner_map_digest", ledger.owner_map_digest},
            {"host_pool_mib", ledger.host_pool_mib},
            {"host_kv_mib", ledger.host_kv_mib},
            {"helpers", std::move(helpers)},
            {"host",
             {{"components", std::move(components)},
              {"available_bytes", ledger.available_bytes},
              {"committed_bytes", ledger.committed_bytes},
              {"remaining_bytes", ledger.remaining_bytes},
              {"pin_available_bytes", ledger.pin_available_bytes},
              {"pin_committed_bytes", ledger.pin_committed_bytes}}}};
}

} // namespace bc::serving::resources
