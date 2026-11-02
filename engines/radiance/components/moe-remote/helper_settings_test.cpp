#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../serving/serving.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
using bc::serving::Json;
namespace fs = std::filesystem;
namespace {
fs::path write(const Json& j) {
    const fs::path p = fs::current_path() / "helper-settings-test-manifest.json";
    std::ofstream out(p);
    if (!out) throw std::runtime_error("test manifest write failed");
    out << j.dump(2) << '\n';
    out.close();
    if (!out) throw std::runtime_error("test manifest write failed");
    return p;
}
void seal(Json& j) {
    j.erase("content_sha256");
    j["content_sha256"] = bc::serving::sha256_hex(j.dump());
}
void must_fail(const Json& j, const std::string& role, const std::string& diagnostic) {
    const auto path = write(j);
    try { (void)bc::serving::load_helper_settings(path, role); }
    catch (const std::exception& ex) {
        if (std::string(ex.what()).find(diagnostic) == std::string::npos)
            throw std::runtime_error("expected " + diagnostic + ", got " + ex.what());
        return;
    }
    throw std::runtime_error("expected failure: " + diagnostic);
}
Json fixture() {
    Json gpu = Json::array();
    const char* arches[] = {"gfx1100", "gfx1100", "gfx1201", "gfx1030"};
    const char* bdfs[] = {"0000:03:00.0", "0000:06:00.0",
                          "0000:09:00.0", "0000:17:00.0"};
    const uint64_t mib[] = {24560, 24560, 32624, 16368};
    for (int i = 0; i < 4; ++i)
        gpu.push_back({{"arch", arches[i]}, {"bdf", bdfs[i]},
                       {"hip_ordinal", i}, {"vram_total_mib", mib[i]}});
    Json facts = {{"host", {{"mem_total_mib", 131072}, {"cpu_count", 20}}},
                  {"gpus", gpu}};
    bc::serving::Options o;
    o.sets = {"engine.model=/model.rad", "engine.radiance_home=/runtime",
              "handoff.heat=/heat.hot-first", "handoff.histogram=off"};
    return bc::serving::resolve(o, facts).manifest;
}
}
int main() {
    const Json base = fixture();
    const auto path = write(base);
    for (unsigned i = 0; i != 3; ++i) {
        auto s = bc::serving::load_helper_settings(path, "helper" + std::to_string(i));
        assert(s.helper.index == i && s.helper.count == 3);
        assert(s.handoff.max_rows == 4096 && s.handoff.small_rows == 8);
        assert(s.handoff.gemv_large_rows == 32 && s.handoff.moe_down_tile == 1);
        assert(s.gemv == 1 && s.gemv_h2 == 2 && s.gemv_w4h2 && s.gemv_wide);
        assert(s.fuseroute && s.vecio && s.sparse_act && s.lane_quant);
        assert(s.native && s.fp16 && s.ahalf);
        assert(s.moe_probe == 0 && !s.refusal_trace);
        assert(s.content_sha256 == base.at("content_sha256").get<std::string>());
        assert(s.budgets_mib.size() == 3 && s.shares.size() == 3);
        if (i == 2) assert(s.rows && s.rows_kernel && s.formats == "w4nl64");
        else assert(s.formats == "w4nl64,fp8_e4m3_block");
    }
    must_fail(base, "helper3", "unknown role");
    must_fail(base, "engine", "unknown role");
    auto changed = base;
    changed["values"]["helper"]["gemv"] = 999;
    must_fail(changed, "helper0", "content_sha256");
    seal(changed);
    must_fail(changed, "helper0", "helper.gemv");
    changed = base;
    changed["values"]["helper"].erase("gemv");
    seal(changed);
    must_fail(changed, "helper0", "missing");
    changed = base;
    changed["values"]["helper"]["not_a_key"] = true;
    seal(changed);
    must_fail(changed, "helper0", "unknown key");
    changed = base;
    changed["process_helpers"][0]["unexpected"] = true;
    seal(changed);
    must_fail(changed, "helper0", "unknown key unexpected");
    // Experiments have an explicit non-production profile, no implicit
    // diagnostic escape hatch through a manifest or a --set.
    const Json facts = {{"host", {{"mem_total_mib", 131072}, {"cpu_count", 20}}},
       {"gpus", Json::array({
          {{"arch","gfx1100"},{"bdf","0000:03:00.0"},{"hip_ordinal",0},{"vram_total_mib",24560}},
          {{"arch","gfx1100"},{"bdf","0000:06:00.0"},{"hip_ordinal",1},{"vram_total_mib",24560}},
          {{"arch","gfx1201"},{"bdf","0000:09:00.0"},{"hip_ordinal",2},{"vram_total_mib",32624}},
          {{"arch","gfx1030"},{"bdf","0000:17:00.0"},{"hip_ordinal",3},{"vram_total_mib",16368}}
       })}};
    bc::serving::Options o;
    o.sets = {"engine.model=/model.rad", "engine.radiance_home=/runtime",
              "handoff.heat=/heat.hot-first", "handoff.histogram=off",
              "experiment.helper.trace=5"};
    try { (void)bc::serving::resolve(o, facts); throw std::runtime_error("production accepted experiment"); }
    catch (const std::runtime_error& e) {
        if (std::string(e.what()) == "production accepted experiment") throw;
        assert(std::string(e.what()).find("rejected") != std::string::npos);
    }
    o.sets.push_back("service.profile=experiment");
    o.sets.push_back("experiment.helper.moe_probe=2");
    o.sets.push_back("experiment.helper.refusal_trace=true");
    o.sets.push_back("helper.native=false");
    o.sets.push_back("helper.fp16=false");
    o.sets.push_back("helper.ahalf=false");
    const auto experiment = bc::serving::resolve(o, facts).manifest;
    assert(bc::serving::load_helper_settings(write(experiment),"helper0").trace == 5);
    const auto configured = bc::serving::load_helper_settings(write(experiment), "helper0");
    assert(configured.moe_probe == 2 && configured.refusal_trace);
    assert(!configured.native && !configured.fp16 && !configured.ahalf);
    o.sets.back() = "experiment.helper.moe_probe=3";
    try { (void)bc::serving::resolve(o, facts); assert(false); }
    catch (const std::runtime_error& e) {
        assert(std::string(e.what()).find("moe_probe") != std::string::npos);
    }
    fs::remove(path);
    return 0;
}
