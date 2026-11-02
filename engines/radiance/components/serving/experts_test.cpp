// bc_experts_test: the GPU-free expert ownership (bc::experts) on a synthetic .rad container: the container reader,
// device sizing, plan_owners against each helper's own select_owned, the owner-map digest (spec §4) and refusals.
#ifdef NDEBUG
#undef NDEBUG   // the checks are assert()s: keep them live in Release component builds
#endif
#include "experts.h"
#include "sha256.h"
#include "test_rad_fixture.h"
#include "golden/golden_cases.h"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;

// spec §4: owner_map_sha256 over "h<i>\n" + "<layer>:<expert>\n" per owned expert, in helper and selection order.
std::string digest(const bc::experts::OwnerMap& m) {
    std::string all;
    for (size_t i = 0; i < m.helpers.size(); ++i) {
        all += "h" + std::to_string(i) + "\n";
        for (const auto& [l, e] : m.helpers[i].expert) all += std::to_string(l) + ":" + std::to_string(e) + "\n";
    }
    return bc::serving::sha256_hex(all);
}
}  // namespace

int main(int argc, char** argv) {
    assert(argc == 2 && "usage: bc_experts_test <golden/owned_golden.txt>");
    const fs::path root = fs::current_path() / "bc-experts-fixture";
    fs::remove_all(root);
    fs::create_directories(root);
    // 4 layers x 16 routed W4NL64 experts, hidden 128, width 128: 25344 bytes an expert on the device.
    const auto model = root / "owners.rad";
    bc::test::write_rad(model, 4, 16, 128, 128, {});
    const auto model_prof = root / "owners-profile.rad";
    bc::test::write_rad(model_prof, 4, 16, 128, 128, bc::test::embedded_profile(4, 16));
    bc::experts::Container fp;
    assert(fp.open(model_prof.c_str()) && fp.profile && fp.h->prof_count == 64);
    bc::experts::Container f;
    assert(f.open(model.c_str()) && f.layers == 4 && f.experts == 16 && f.hidden == 128 && f.intermediate == 128);
    assert(bc::experts::laid_out_bytes(f, true, 0, 0) == 25344);
    // A heat ranking: uint32[128 layers][16 experts], a different order in every layer.
    std::vector<uint32_t> heat(128 * 16, 0);
    for (uint32_t l = 0; l < 4; ++l)
        for (uint32_t e = 0; e < 16; ++e) heat[l * 16 + e] = (e * 7 + l * 3) % 16 + 1;
    const auto heat_path = root / "owners.hot-first";
    {
        std::ofstream h(heat_path, std::ios::binary);
        h.write(reinterpret_cast<const char*>(heat.data()), static_cast<std::streamsize>(heat.size() * 4));
    }
    auto policy = [&](const char* formats) {
        bc::experts::OwnerPolicy p;
        p.heat = heat_path.string();
        p.placement = "tail";
        p.formats = formats;
        p.shares = "1,1,1";
        p.balance_speeds = {1.0, 1.0};
        return p;
    };
    const std::vector<bc::experts::OwnerPolicy> pols = {policy("w4nl64,fp8_e4m3_block"), policy("w4nl64,fp8_e4m3_block"),
                                                        policy("w4nl64")};
    const std::vector<std::string> archs = {"gfx1100", "gfx1100", "gfx1030"};
    const uint64_t e = 25344;
    bc::experts::OwnerMap map;
    std::string why;
    // Two heads of 20 experts each, a tail of 10: 50 of 64 owned, 14 left to the engine.
    assert(bc::experts::plan_owners(f, pols, archs, {20 * e, 20 * e, 10 * e}, &map, &why));
    assert(map.routed == 64 && map.helpers.size() == 3);
    assert(map.helpers[0].expert.size() == 20 && map.helpers[1].expert.size() == 20 && map.helpers[2].expert.size() == 10);
    assert(map.helpers[2].bytes == 10 * e && map.unowned.size() == 14);
    // Each helper's own selection (as the helper process runs it) is exactly its part of the map.
    uint64_t b[3] = {20 * e, 20 * e, 10 * e};
    for (uint32_t h = 0; h < 3; ++h) {
        bc::experts::Owned own;
        assert(bc::experts::select_owned(f, pols[h], archs[h], true, {}, b, h, 3, &own));
        assert(own.expert == map.helpers[h].expert && own.bytes == map.helpers[h].bytes);
    }
    // The digest is stable, and any change of budget changes it.
    bc::experts::OwnerMap again;
    assert(bc::experts::plan_owners(f, pols, archs, {20 * e, 20 * e, 10 * e}, &again, &why));
    assert(digest(again) == digest(map) && digest(map).size() == 64);
    bc::experts::OwnerMap bigger_tail;
    assert(bc::experts::plan_owners(f, pols, archs, {20 * e, 20 * e, 14 * e}, &bigger_tail, &why));
    assert(bigger_tail.helpers[2].expert.size() == 14 && bigger_tail.unowned.size() == 10);
    assert(digest(bigger_tail) != digest(map));
    // The tail card runs only w4nl64; an fp8-only formats list for it is refused, naming the helper.
    auto refused = pols;
    refused[2].formats = "fp8_e4m3_block";
    assert(!bc::experts::plan_owners(f, refused, archs, {20 * e, 20 * e, 10 * e}, &map, &why));
    assert(why.find("helper 2 (gfx1030)") != std::string::npos);
    assert(!bc::experts::plan_owners(f, pols, archs, {20 * e, 20 * e}, &map, &why));
    assert(why.find("one policy, card and budget per helper") != std::string::npos);
    // The same cases the pre-move worker selection ran (golden/README.md): every helper's owned set, line for line.
    {
        std::string got;
        for (const golden::Case& c : golden::cases()) {
            bc::experts::OwnerPolicy p;
            p.formats = "w4nl64";
            p.balance_speeds = {1.0, 1.0};
            for (const auto& [k, v] : c.env) {
                const std::string key = k;
                if (key == "HEAT") p.heat = v == "@heat" ? heat_path.string() : v;
                else if (key == "PLACEMENT") p.placement = v;
                else if (key == "FORMATS") p.formats = v;
                else if (key == "SHARES") p.shares = v;
                else if (key == "BALANCE") p.balance = !v.empty() && v != "0";
                else if (key == "BALANCE_SPEED") {
                    double a = 0.0, b = 0.0;
                    if (std::sscanf(v.c_str(), "%lf,%lf", &a, &b) == 2 && a > 0.0 && b > 0.0) p.balance_speeds = {a, b};
                }
            }
            std::vector<uint64_t> budgets;
            for (uint64_t n : c.budget_experts) budgets.push_back(n * golden::kExpertBytes);
            for (uint32_t i = 0; i < c.helpers; ++i) {
                bc::experts::Owned o;
                const bool ok = bc::experts::select_owned(c.embedded_profile ? fp : f, p, "host", false, {}, budgets.data(), i, c.helpers, &o);
                got += std::string(c.name) + " " + std::to_string(i) + (ok ? " ok " : " refused ") + std::to_string(ok ? o.bytes : 0);
                if (ok) for (auto [l, e] : o.expert) got += " " + std::to_string(l) + ":" + std::to_string(e);
                got += "\n";
            }
        }
        std::ifstream gin(argv[1]);
        std::ostringstream want;
        want << gin.rdbuf();
        if (got != want.str()) {
            std::ofstream(root / "owned_got.txt") << got;
            std::fprintf(stderr, "owned sets differ from %s (ours in %s)\n", argv[1], (root / "owned_got.txt").c_str());
        }
        assert(got == want.str());
    }
    // Policies the worker's environment parser would never build are refused at the library boundary, never thrown on or truncated.
    {
        bc::experts::Owned own;
        const uint64_t two[2] = {20 * e, 20 * e};
        bc::experts::OwnerPolicy bal = policy("w4nl64");
        bal.placement = "roundrobin";
        bal.balance = true;
        // balance mode does not read the shares, as on main: not-all-zero shares are not required there
        bal.shares = "0,0";
        assert(bc::experts::select_owned(f, bal, "gfx1100", true, {}, two, 0, 2, &own) && !own.expert.empty());
        // without balance the same shares are refused
        bc::experts::OwnerPolicy plain = bal;
        plain.balance = false;
        assert(!bc::experts::select_owned(f, plain, "gfx1100", true, {}, two, 0, 2, &own));
        plain.shares = "4294967296,1";
        assert(!bc::experts::select_owned(f, plain, "gfx1100", true, {}, two, 0, 2, &own));
        // balance with no or one speed is refused
        bal.balance_speeds.clear();
        assert(!bc::experts::select_owned(f, bal, "gfx1100", true, {}, two, 0, 2, &own));
        bal.balance_speeds = {1.0};
        assert(!bc::experts::select_owned(f, bal, "gfx1100", true, {}, two, 0, 2, &own));
        // error order: no budget, then backend, then the layout check, then the shares
        bool layout_called = false;
        const auto layout = [&] { layout_called = true; return false; };
        const uint64_t none[2] = {0, 20 * e};
        plain.shares = "0,0";
        assert(!bc::experts::select_owned(f, plain, "gfx1100", true, layout, none, 0, 2, &own) && !layout_called);
        assert(!bc::experts::select_owned(f, plain, "gfx1100", true, layout, two, 0, 2, &own) && layout_called);
        // more helpers than the runtime protocol holds
        std::vector<bc::experts::OwnerPolicy> many(r3::moe_shm::kMaxHelpers + 1, policy("w4nl64"));
        assert(!bc::experts::plan_owners(f, many, std::vector<std::string>(many.size(), "gfx1100"),
                                         std::vector<uint64_t>(many.size(), 4 * e), &map, &why));
    }
    fs::remove_all(root);
    std::printf("ok\n");
    return 0;
}
