// bc_owner_map_test: verify_owner_map against the shared planner on a synthetic .rad container (RSC08 slice S1).
// The manifest under test is the production example with its placement pointed at the fixture; its owner map is written
// from bc::experts::plan_owners with a hand-built policy, so the check is not the planner compared with itself: the
// policy the manifest implies (owner_policies) must also equal the hand-built one field by field.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "owner_map.h"
#include "serving_codec.h"
#include "sha256.h"
#include "test_rad_fixture.h"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace {
namespace fs = std::filesystem;
namespace v1 = bc::serving::v1;
using Json = nlohmann::json;

std::string read(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

v1::Manifest decode_json(const Json& j) {
    v1::Manifest m;
    const v1::Status e = v1::decode(j, "", &m);
    if (e) {
        std::fprintf(stderr, "decode: %s %s\n", e->path.c_str(), e->message.c_str());
        assert(false);
    }
    return m;
}

bool refused_at(const v1::Manifest& m, const bc::experts::Container& model, const std::string& path) {
    const v1::Status s = bc::serving::verify_owner_map(m, model);
    if (!s || s->code != v1::ConfigErrorCode::ownership_mismatch) return false;
    if (s->path.find(path) != 0) std::fprintf(stderr, "wanted %s, got %s: %s\n", path.c_str(), s->path.c_str(), s->message.c_str());
    return s->path.find(path) == 0;
}
}  // namespace

int main(int, char**) {
    const fs::path root = fs::current_path() / "bc-owner-map-fixture";
    fs::remove_all(root);
    fs::create_directories(root);
    const auto model_path = root / "owners.rad";
    bc::test::write_rad(model_path, 4, 16, 512, 512, bc::test::embedded_profile(4, 16));
    bc::experts::Container model;
    assert(model.open(model_path.c_str()) && model.layers == 4 && model.experts == 16);
    const fs::path heat = root / "owners.hot-first";   // present so the placement is the declared one, not a fallback
    {
        std::vector<uint32_t> counts(128 * 16, 0);
        for (uint32_t l = 0; l < 4; ++l)
            for (uint32_t e = 0; e < 16; ++e) counts[l * 16 + e] = (e * 7 + l * 3) % 16 + 1;
        std::ofstream(heat, std::ios::binary).write(reinterpret_cast<const char*>(counts.data()), counts.size() * 4);
    }

    // the production example, its placement pointed at the fixture: tail strategy, shares 1/1/0, formats per helper
    Json j = Json::parse(read(fs::path(BC_CONTRACT_DIR) / "example.flashnext-4gpu.json"));
    j["spec"]["handoff"]["placement"]["heat_file"] = heat.string();
    const int64_t mib[3] = {10, 10, 5};
    for (int i = 0; i < 3; ++i) j["spec"]["processes"]["helpers"][i]["vram_budget_mib"] = mib[i];

    // the owner map the planner produces for a hand-built policy
    auto policy = [&](const char* formats) {
        bc::experts::OwnerPolicy p;
        p.heat = heat.string();
        p.placement = "tail";
        p.formats = formats;
        p.shares = "1,1,0";
        p.balance_speeds = {1.0, 1.0};
        return p;
    };
    const std::vector<bc::experts::OwnerPolicy> hand = {policy("w4nl64,fp8_e4m3_block"), policy("w4nl64,fp8_e4m3_block"), policy("w4nl64")};
    const std::vector<std::string> archs = {"gfx1100", "gfx1100", "gfx1030"};
    const std::vector<uint64_t> budgets = {10ull << 20, 10ull << 20, 5ull << 20};
    bc::experts::OwnerMap planned;
    std::string why;
    assert(bc::experts::plan_owners(model, hand, archs, budgets, &planned, &why));
    assert(planned.helpers[0].expert.size() > 0 && planned.helpers[2].expert.size() > 0);

    Json helpers = Json::array();
    std::string all;
    for (size_t i = 0; i < planned.helpers.size(); ++i) {
        Json experts = Json::array();
        std::vector<std::array<int64_t, 2>> pairs;
        for (auto [l, e] : planned.helpers[i].expert) {
            experts.push_back(Json::array({l, e}));
            pairs.push_back({static_cast<int64_t>(l), static_cast<int64_t>(e)});
        }
        const std::string segment = v1::owner_segment(i, pairs);
        all += segment;
        helpers.push_back({{"owned_sha256", bc::serving::sha256_hex(segment)}, {"count", experts.size()},
                           {"bytes", planned.helpers[i].bytes}, {"experts", experts}});
    }
    j["spec"]["plan"]["owner_map"] = {{"owner_map_sha256", bc::serving::sha256_hex(all)}, {"helpers", helpers}};
    const v1::Manifest good = decode_json(j);

    // the policies the manifest implies are the hand-built ones, field by field
    const auto implied = bc::serving::owner_policies(good);
    assert(implied.size() == 3);
    for (size_t i = 0; i < 3; ++i)
        assert(implied[i].heat == hand[i].heat && implied[i].placement == hand[i].placement && implied[i].formats == hand[i].formats &&
               implied[i].shares == hand[i].shares && implied[i].balance == hand[i].balance);
    assert(bc::serving::helper_archs(good) == archs && bc::serving::helper_budgets(good) == budgets);

    // an owner map equal to the planner's is accepted
    assert(!bc::serving::verify_owner_map(good, model));

    // exact bytes: one byte more or less is refused at the helper's bytes pointer
    for (int64_t delta : {1, -1}) {
        v1::Manifest m = good;
        m.spec.plan.owner_map.helpers[1].bytes += delta;
        assert(refused_at(m, model, "spec.plan.owner_map.helpers[1].bytes"));
    }
    // bytes that the experts cannot take at all (the reader only refused bytes contradicting emptiness)
    {
        v1::Manifest m = good;
        m.spec.plan.owner_map.helpers[0].bytes = 1;
        assert(refused_at(m, model, "spec.plan.owner_map.helpers[0].bytes"));
    }
    // a different expert at a position, and a swap of two positions (the order is part of the selection)
    {
        v1::Manifest m = good;
        std::swap(m.spec.plan.owner_map.helpers[0].experts[0], m.spec.plan.owner_map.helpers[0].experts[1]);
        assert(refused_at(m, model, "spec.plan.owner_map.helpers[0].experts[0]"));
    }
    // one expert fewer, with its count kept consistent
    {
        v1::Manifest m = good;
        m.spec.plan.owner_map.helpers[2].experts.pop_back();
        m.spec.plan.owner_map.helpers[2].count -= 1;
        assert(refused_at(m, model, "spec.plan.owner_map.helpers[2].experts"));
    }
    // the declared placement decides the map: a different share, strategy or heat file gives a different selection
    {
        v1::Manifest m = good;
        m.spec.processes.helpers[0].share = 3;
        assert(refused_at(m, model, "spec.plan.owner_map.helpers[") );
    }
    {
        v1::Manifest m = good;
        m.spec.handoff.placement.strategy = v1::PlacementStrategy::kRoundrobin;
        assert(refused_at(m, model, "spec.plan.owner_map.helpers["));
    }
    // a placement the planner refuses (a format the tail card cannot run) names the owner map
    {
        v1::Manifest m = good;
        m.spec.processes.helpers[2].formats = {v1::ExpertFormat::kFp8E4m3Block};
        assert(refused_at(m, model, "spec.plan.owner_map"));
    }
    fs::remove_all(root);
    std::printf("ok\n");
    return 0;
}
