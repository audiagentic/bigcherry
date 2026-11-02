// Card-free test of rad0011's start-up plan check (core/bc_plan_memory_check.cpp): a pure function of two plan
// documents. Built and run by tools/tests/patch/test_radiance_patch_packages.py. Prints "ok" on success.
#include "bc_plan_memory_check.h"

#include <cstdio>
#include <string>

namespace {
int failures = 0;

nlohmann::json base() {
    return nlohmann::json::parse(R"json({
      "plan_version": 1, "producer": "radiance --plan-memory (rad0011)",
      "host": [{"name": "ngram_table", "bytes": 1000, "residence": "pinned", "from": "engine plan (rad0011: weight_footprint)",
                "remedy": "set engine.ngram_placement = \"disk\""}],
      "checkpoints": {"slot_bytes": 100, "floor_slots": 8, "want_slots": 72, "from": "engine plan (rad0011: ckpt)"},
      "kv_one_session": {"bytes": 500, "from": "engine plan (rad0011: kv_one)"},
      "vram": {"claimable": 9000, "statics": 3000, "elastic": 6000, "experts": 0, "kv": 5000, "stage": 200,
               "expert_floor": 0, "kv_ceiling": 7000, "expert_ceiling": 100, "from": "engine plan (rad0011: vram)",
               "cards": [{"capacity": 10000, "free": 9900, "engine_expert_plane": 100, "expert_unit_max": 10}]},
      "host_pool_mib": 9216})json");
}

void expect(bool want, const nlohmann::json& live, const std::string& needle) {
    std::string why;
    const bool got = bc_plan_memory_check(base(), live, &why);
    if (got != want || (!want && why.find(needle) == std::string::npos)) {
        std::printf("FAIL: want %d got %d for '%s': %s\n", want, got, needle.c_str(), why.c_str());
        ++failures;
    }
}
}  // namespace

int main() {
    expect(true, base(), "");
    nlohmann::json live = base();
    live["host"][0]["bytes"] = 999;   // smaller is fine
    live["vram"]["claimable"] = 9500; // more room is fine
    expect(true, live, "");
    live = base(); live["host"][0]["bytes"] = 1001;
    expect(false, live, "host consumer ngram_table: live 1001 bytes exceed the planned 1000 bytes (engine plan (rad0011: weight_footprint)); set engine.ngram_placement = \"disk\"");
    live = base(); live["host"].push_back({{"name", "embedding"}, {"bytes", 1}, {"residence", "pinned"},
                                           {"from", "engine plan (rad0011: emb)"}, {"remedy", ""}});
    expect(false, live, "host consumer embedding: live 1 bytes exceed the planned 0 bytes");
    expect(false, live, "bc-serve --plan-memory");
    live = base(); live["checkpoints"]["slot_bytes"] = 101;
    expect(false, live, "checkpoint slot: live 101 bytes exceed the planned 100");
    live = base(); live["checkpoints"]["floor_slots"] = 9;
    expect(false, live, "checkpoint floor: live 900 bytes exceed the planned 800");
    live = base(); live["kv_one_session"]["bytes"] = 501;
    expect(false, live, "one full-length session's KV: live 501");
    live = base(); live["vram"]["statics"] = 3001;
    expect(false, live, "VRAM statics: live 3001");
    live = base(); live["vram"]["stage"] = 201;
    expect(false, live, "VRAM stage: live 201");
    live = base(); live["vram"]["claimable"] = 9000 - 100;    // a live reading that moved a little: the split still fits
    expect(true, live, "");
    live = base(); live["vram"]["claimable"] = 8199;           // the planned split is 3000 + 5000 + 0 + 200 = 8200
    expect(false, live, "claimable VRAM (for the planned split: statics + kv + experts + stage): live 8199 bytes fall short of the planned 8200 bytes");
    live = base(); live["vram"]["cards"][0]["capacity"] = 9999;
    expect(false, live, "card 0 capacity: live 9999 bytes fall short of the planned 10000");
    live = base(); live["vram"]["cards"].push_back(live["vram"]["cards"][0]);
    expect(false, live, "the live plan has 2 card(s), the manifest's 1");
    live = base(); live["plan_version"] = 2;
    expect(false, live, "plan_version 1");
    if (failures) return 1;
    std::printf("ok\n");
    return 0;
}
