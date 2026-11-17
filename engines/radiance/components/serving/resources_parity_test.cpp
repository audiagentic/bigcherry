// RSC03 admission invariants against checked-in Python-oracle observations.
// Fixture is external JSON; the native calculation accepts only typed measured facts.
#include "resources.h"

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#ifndef BC_RSC02_PARITY_FIXTURE
#error "BC_RSC02_PARITY_FIXTURE must name golden/rsc02_resource_parity.json"
#endif

namespace {
using Json = nlohmann::json;
namespace resources = bc::serving::resources;
constexpr uint64_t MiB = uint64_t{1024} * 1024;

resources::ResourceRequest fixture_request(const Json& input) {
    resources::ResourceRequest request;
    request.model = {
        input.at("owner_map_digest").get<std::string>(),  input.at("embedding_bytes").get<uint64_t>(),
        input.at("unowned_expert_bytes").get<uint64_t>(), input.at("staging_bytes").get<uint64_t>(),
        input.at("ngram_ram_bytes").get<uint64_t>(),      input.at("checkpoint_bytes").get<uint64_t>(),
        input.at("other_engine_bytes").get<uint64_t>(),   input.at("kv_bytes_per_token").get<uint64_t>()};
    request.target = {
        input.at("context_tokens").get<uint64_t>(), input.at("concurrent_sessions").get<uint64_t>(),
        input.at("retained_tokens").get<uint64_t>(), input.at("margin_mib").get<uint64_t>() * MiB,
        input.at("reserve_mib").get<uint64_t>() * MiB};
    request.host = {input.at("available_mib").get<uint64_t>() * MiB,
                    input.at("cgroup_room_mib").get<uint64_t>() * MiB,
                    input.at("pin_room_mib").get<uint64_t>() * MiB};
    for (const Json& helper : input.at("helpers")) {
        request.helpers.push_back({helper.at("bdf").get<std::string>(), helper.at("arch").get<std::string>(),
                                   helper.at("vram_mib").get<uint64_t>(),
                                   helper.at("reserve_mib").get<uint64_t>(),
                                   helper.at("worker_bytes").get<uint64_t>(), std::nullopt});
    }
    return request;
}

void refuses(const resources::ResourceRequest& request, bc::serving::v2::ConfigErrorCode code,
             const std::string& phrase) {
    const auto result = resources::calculate(request);
    if (result.ok() || result.error().code != code ||
        result.error().message.find(phrase) == std::string::npos) {
        std::cerr << "expected resource refusal containing '" << phrase << "'\n";
        if (!result.ok())
            std::cerr << result.error().message << '\n';
        assert(false);
    }
}
} // namespace

int main() {
    try {
        std::ifstream file(BC_RSC02_PARITY_FIXTURE);
        if (!file) {
            std::cerr << "missing Python resource oracle\n";
            return 1;
        }
        const Json fixture = Json::parse(file);
        const Json& cases = fixture.at("cases");
        assert(cases.is_array() && cases.size() == 32);
        bool flash = false, qwen = false, tail = false, dense = false, kv = false;
        for (const Json& row : cases) {
            const std::string name = row.at("name").get<std::string>();
            const resources::ResourceRequest request = fixture_request(row.at("input"));
            const auto actual = resources::calculate(request);
            if (!actual.ok()) {
                std::cerr << name << ": native admission refused a golden case: " << actual.error().message
                          << '\n';
                return 1;
            }
            const Json received = resources::to_json(actual.value());
            if (received != row.at("expected")) {
                std::cerr << name << ": C++ and Python resource derivation differ\n"
                          << "C++ " << received.dump(2) << "\nPython " << row.at("expected").dump(2) << '\n';
                return 1;
            }
            flash |= name.rfind("flashnext-", 0) == 0;
            qwen |= name.rfind("qwen27b-", 0) == 0;
            tail |= name == "flashnext-4gpu" && request.helpers.size() == 3;
            dense |= request.helpers.empty();
            kv |= actual.value().host_kv_mib > 0;
        }
        assert(flash && qwen && tail && dense && kv);
        const resources::ResourceRequest base = fixture_request(cases.at(0).at("input"));
        auto sample = base;
        sample.model.owner_map_digest = "not-a-digest";
        refuses(sample, bc::serving::v2::ConfigErrorCode::invalid_value, "owner_map_digest");
        sample = base;
        sample.host.cgroup_headroom_bytes = MiB;
        refuses(sample, bc::serving::v2::ConfigErrorCode::admission_refused, "host RAM admission");
        sample = base;
        sample.host.pin_available_bytes = MiB;
        refuses(sample, bc::serving::v2::ConfigErrorCode::admission_refused, "pin admission");
        sample = base;
        assert(!sample.helpers.empty());
        sample.helpers.at(0).worker_footprint_bytes = 0;
        refuses(sample, bc::serving::v2::ConfigErrorCode::invalid_value, "must be positive");
        sample = base;
        sample.helpers.push_back(sample.helpers.at(0));
        refuses(sample, bc::serving::v2::ConfigErrorCode::invalid_value, "duplicate helper");
        std::cout << "RSC02: " << cases.size()
                  << " C++/Python config/profile derivations and refusals passed\n";
    } catch (const std::exception& error) {
        std::cerr << "resource oracle fixture rejected: " << error.what() << '\n';
        return 1;
    }
}
