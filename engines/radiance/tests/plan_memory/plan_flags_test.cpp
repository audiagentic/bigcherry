// Card-free test of rad0011's plan mode and start-up check (core/bc_plan_check.cpp): --plan-memory FILE, the check armed
// with an expected plan and floor (from a measured manifest in the engine; from the admission vectors here), and the one
// start-up line. The manifest-to-check path through main()'s real call order is manifest_main_order_test.cpp. Built and run by tools/tests/patch/test_radiance_patch_packages.py with the admission
// vectors and a scratch directory as arguments. Prints "ok" on success.
#include "bc_plan_check.h"

#include "admission.h"
#include "serving_codec.h"
#include "strict_json.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

// rad0011 reads the manifest reader's expected plan (rad0009 core/bc_serving_manifest.cpp, linked here); it asks the engine's
// option table for its names, which this test never uses.
namespace rad::diag {
std::vector<std::string> config_option_names() { return {}; }
}  // namespace rad::diag

namespace {
namespace v1 = bc::serving::v1;
using Json = nlohmann::json;
int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what.c_str());
        ++failures;
    }
}

/* Runs fn with stderr redirected to a file and returns what it wrote. */
std::string captured(const std::string& scratch, const std::function<void()>& fn) {
    const std::string file = scratch + "/stderr.txt";
    std::fflush(stderr);
    std::freopen(file.c_str(), "w", stderr);
    fn();
    std::fflush(stderr);
    std::ifstream in(file);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

int take(const std::string& scratch, std::vector<std::string> args, std::string* err, std::vector<std::string>* left = nullptr) {
    std::vector<char*> argv = {const_cast<char*>("radiance")};
    for (std::string& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    int argc = static_cast<int>(argv.size()) - 1;
    char** av = argv.data();
    int rc = -1;
    *err = captured(scratch, [&] { rc = bc_plan::take_flags(&argc, &av); });
    if (left) {
        left->clear();
        for (int i = 1; i < argc; ++i) left->push_back(av[i]);
    }
    return rc;
}

void write(const std::string& path, const std::string& text) {
    std::ofstream(path, std::ios::binary | std::ios::trunc) << text;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::printf("usage: plan_flags_test <admission-vectors.json> <scratch dir>\n");
        return 2;
    }
    const std::string scratch = argv[2];
    std::ifstream vin(argv[1]);
    const Json vectors = Json::parse(vin);
    const Json* prod = nullptr;
    for (const Json& v : vectors.at("vectors"))
        if (v.at("name") == "prod-a3-reading-1") prod = &v;
    if (!prod) {
        std::printf("FAIL: no prod-a3-reading-1 vector\n");
        return 1;
    }
    const std::string expected_file = scratch + "/expected.plan.json";   // only to show the retired flag is not taken
    write(expected_file, prod->at("expected").dump());
    const std::string ppm = "600000";
    std::string err;
    std::vector<std::string> left;
    int rc0 = -1;

    // no flags: nothing is taken, nothing changes
    check(take(scratch, {"--model", "m.rad", "--port", "8080"}, &err, &left) == 0 && bc_plan::mode() == bc_plan::Mode::none &&
              left == std::vector<std::string>({"--model", "m.rad", "--port", "8080"}) && err.empty(),
          "no plan flags: argv untouched, mode none");

    // --plan-memory FILE is taken off argv
    check(take(scratch, {"--model", "m.rad", "--plan-memory", "/tmp/out.json", "--port", "1"}, &err, &left) == 0 &&
              bc_plan::mode() == bc_plan::Mode::measure && bc_plan::measure_out() == "/tmp/out.json" &&
              left == std::vector<std::string>({"--model", "m.rad", "--port", "1"}),
          "--plan-memory takes its FILE and leaves the rest");
    check(take(scratch, {"--plan-memory"}, &err) == 2 && err.find("BC_CFG_INVALID_VALUE") != std::string::npos &&
              err.find("needs a value") != std::string::npos, "--plan-memory without a file is a usage error: " + err);

    // --plan-memory twice is a usage error; there is no flag that supplies an expected plan
    check(take(scratch, {"--plan-memory", "/tmp/a.json", "--plan-memory", "/tmp/b.json"}, &err) == 2 &&
              err.find("given twice") != std::string::npos, "a repeated --plan-memory is a usage error: " + err);
    check(take(scratch, {"--plan-check", expected_file, "--plan-check-floor-ppm", ppm}, &err, &left) == 0 &&
              bc_plan::mode() == bc_plan::Mode::none &&
              left == std::vector<std::string>({"--plan-check", expected_file, "--plan-check-floor-ppm", ppm}),
          "the retired --plan-check flags are not taken: they stay on argv for radiance to refuse as unknown options");

    // the floor is range-checked by the shared validator when the check is armed
    v1::EnginePlan expected_plan;
    check(!v1::decode(prod->at("expected"), "", &expected_plan), "the vector's expected plan decodes");
    for (int64_t bad : {int64_t(-1), int64_t(1000000), int64_t(5000000)}) {
        err = captured(scratch, [&] { rc0 = bc_plan::arm(expected_plan, bad); });
        check(rc0 == 2 && err.find("kv_backing_floor_ppm must be in [0, 1000000)") != std::string::npos &&
                  bc_plan::mode() != bc_plan::Mode::check,
              "floor " + std::to_string(bad) + " is refused and arms nothing: rc " + std::to_string(rc0) + " " + err);
    }
    check(bc_plan::arm(expected_plan, 999999) == 0 && bc_plan::mode() == bc_plan::Mode::check, "floor 999999 is in range");
    check(bc_plan::arm(expected_plan, 0) == 0, "floor 0 is in range");

    // the check: admitted prints one line with its margins; a refusal is the shared error with its exit code
    check(bc_plan::arm(expected_plan, 600000) == 0 && bc_plan::mode() == bc_plan::Mode::check, "a valid check is armed");
    v1::EnginePlan live;
    check(!v1::decode(prod->at("live"), "", &live), "the vector's live plan decodes");
    int rc = -1;
    err = captured(scratch, [&] { rc = bc_plan::check(live); });
    check(rc == 0 && err.find("plan check: admitted") != std::string::npos && err.find("margin") != std::string::npos &&
              std::count(err.begin(), err.end(), '\n') == 1,
          "an admitted plan prints one line with margins: " + err);

    v1::EnginePlan short_kv = live;
    short_kv.vram.kv_resolved = 1;   // far below kv_required
    err = captured(scratch, [&] { rc = bc_plan::check(short_kv); });
    check(rc == 4 && err.find("radiance: config: BC_CFG_ADMISSION_REFUSED (RAD_E_NOMEM, -6)") != std::string::npos &&
              err.find("admission: resolved KV") != std::string::npos && err.find("kv_required") != std::string::npos,
          "a short resolved KV is refused, exit 4, naming kv_required: " + err);

    v1::EnginePlan drifted = live;
    drifted.vram.statics += 1;   // a stable fact that moved
    err = captured(scratch, [&] { rc = bc_plan::check(drifted); });
    check(rc == 2 && err.find("BC_CFG_PLAN_MISMATCH") != std::string::npos && err.find("VRAM statics") != std::string::npos,
          "a drifted stable fact is a plan mismatch, exit 2: " + err);

    // one byte of drift in a host consumer is refused too
    v1::EnginePlan host_drift = live;
    host_drift.host[0].bytes += 1;
    err = captured(scratch, [&] { rc = bc_plan::check(host_drift); });
    check(rc == 2 && err.find("BC_CFG_PLAN_MISMATCH") != std::string::npos, "one byte of host drift is refused: " + err);

    // Every normative admission vector through the engine's own entry points: the expected plan decoded as the manifest
    // reader decodes it, bc_plan::arm (the floor), then bc_plan::check. The exit code and the message must be what the vector
    // says (admit 0, admission 4, plan or input 2).
    {
        // Where each vector meets production code: 54 reach bc_plan::check; the rest are refused earlier, at the strict decode
        // of the expected plan or the floor range (3), or are a live plan the contract cannot even decode (1: the engine builds
        // its live plan in process, as a typed struct, so a missing field there is a decode failure of the vector).
        int ran = 0, through_check = 0, refused_at_flags = 0, live_undecodable = 0;
        for (const Json& v : vectors.at("vectors")) {
            const std::string name = v.at("name").get<std::string>(), cls = v.at("verdict").value("class", std::string("admit"));
            const bool admit = v.at("verdict").at("admit").get<bool>();
            const int want_exit = admit ? 0 : cls == "admission" ? 4 : 2;
            const std::string needle = admit ? std::string() : v.at("verdict").at("needle").get<std::string>();
            ++ran;
            const int64_t vppm = v.at("ppm").get<int64_t>();
            v1::EnginePlan expected, live_plan;
            // the strict reader first (it is what refuses a byte count past I-JSON's 2^53 - 1), then the generated decode
            v1::Result<Json> strict = v1::parse_strict_json(v.at("expected").dump(), "vec-expected.json");
            v1::Status decoded = strict ? v1::decode(strict.value(), "", &expected) : v1::Status{};
            int rc = 0;
            err = captured(scratch, [&] {
                if (!strict) {
                    std::fputs(v1::render_human("radiance", strict.error()).c_str(), stderr);
                    rc = v1::exit_code_of(strict.error());
                } else if (decoded) {
                    std::fputs(v1::render_human("radiance", *decoded).c_str(), stderr);
                    rc = v1::exit_code_of(*decoded);
                } else {
                    rc = bc_plan::arm(expected, vppm);
                }
            });
            const bool live_ok = !v1::decode(v.at("live"), "", &live_plan);
            if (rc != 0) {
                ++refused_at_flags;
                check(!admit && cls == "input" && rc == 2, "vector " + name + ": refused before the check with exit 2, got " + std::to_string(rc) + " " + err);
                if (vppm < 0 || vppm >= 1000000)
                    check(err.find("kv_backing_floor_ppm must be in [0, 1000000)") != std::string::npos && err.find("BC_CFG_INVALID_VALUE") != std::string::npos,
                          "vector " + name + ": the floor refusal names the range: " + err);
                else
                    check(err.find("BC_CFG_INVALID_FORMAT") != std::string::npos && err.find("vram.kv_ceiling") != std::string::npos &&
                              err.find("outside I-JSON's +/-(2^53 - 1)") != std::string::npos && err.find("vec-expected.json") != std::string::npos,
                          "vector " + name + ": the expected-plan refusal is invalid_format at vram.kv_ceiling naming the file: " + err);
                continue;
            }
            check(bc_plan::mode() == bc_plan::Mode::check, "vector " + name + ": the check is armed");
            if (!live_ok) {
                ++live_undecodable;
                const auto e = v1::decode(v.at("live"), "live", &live_plan);
                check(!admit && cls == "input" && e && e->code == (name == "kv-resolved-missing" ? v1::ConfigErrorCode::missing_field
                                                                                                   : v1::ConfigErrorCode::invalid_value),
                      "vector " + name + ": the live plan fails decoding with the vector's code");
                continue;
            }
            ++through_check;
            err = captured(scratch, [&] { rc = bc_plan::check(live_plan); });
            check(rc == want_exit, "vector " + name + ": exit " + std::to_string(want_exit) + ", got " + std::to_string(rc) + " " + err);
            if (!admit)
                check(err.find(needle) != std::string::npos, "vector " + name + ": message names '" + needle + "': " + err);
            else
                check(err.find("plan check: admitted") != std::string::npos, "vector " + name + ": admitted line: " + err);
        }
        check(ran >= 46 && through_check + refused_at_flags + live_undecodable == ran, "every vector met production code");
        check(through_check == 54 && refused_at_flags == 3 && live_undecodable == 1,
              "54 through bc_plan::check, 3 refused before it, 1 undecodable live plan (got " + std::to_string(through_check) + "/" +
                  std::to_string(refused_at_flags) + "/" + std::to_string(live_undecodable) + ")");
    }

    std::fclose(stderr);
    if (failures) return 1;
    std::printf("ok\n");
    return 0;
}
