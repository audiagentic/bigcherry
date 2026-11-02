// Card-free test of rad0011's plan flags and start-up check (core/bc_plan_check.cpp): --plan-memory FILE, and
// --plan-check FILE with --plan-check-floor-ppm N (both or neither), the expected plan read through the shared v1 codec,
// and the one start-up line. Built and run by tools/tests/patch/test_radiance_patch_packages.py with the admission
// vectors and a scratch directory as arguments. Prints "ok" on success.
#include "bc_plan_check.h"

#include "admission.h"
#include "serving_codec.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

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
    const std::string expected_file = scratch + "/expected.plan.json";
    write(expected_file, prod->at("expected").dump());
    const std::string ppm = "600000";
    std::string err;
    std::vector<std::string> left;

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

    // --plan-check and --plan-check-floor-ppm go together
    check(take(scratch, {"--plan-check", expected_file}, &err) == 2 && err.find("go together") != std::string::npos,
          "--plan-check without the floor is a usage error: " + err);
    check(take(scratch, {"--plan-check-floor-ppm", ppm}, &err) == 2 && err.find("go together") != std::string::npos,
          "the floor without --plan-check is a usage error: " + err);
    check(take(scratch, {"--plan-memory", "/tmp/o.json", "--plan-check", expected_file, "--plan-check-floor-ppm", ppm}, &err) == 2 &&
              err.find("does not go with") != std::string::npos,
          "--plan-memory does not go with --plan-check: " + err);
    check(take(scratch, {"--plan-check", expected_file, "--plan-check", expected_file, "--plan-check-floor-ppm", ppm}, &err) == 2 &&
              err.find("given twice") != std::string::npos, "a repeated flag is a usage error: " + err);

    // the floor is range-checked by the shared validator
    for (const char* bad : {"-1", "1000000", "abc", "", "5x"}) {
        const int rc = take(scratch, {"--plan-check", expected_file, "--plan-check-floor-ppm", bad}, &err);
        check(rc == 2 && (err.find("kv_backing_floor_ppm must be in [0, 1000000)") != std::string::npos ||
                          err.find("must be an integer") != std::string::npos || err.find("needs a value") != std::string::npos),
              std::string("floor '") + bad + "' is refused: rc " + std::to_string(rc) + " " + err);
    }
    check(take(scratch, {"--plan-check", expected_file, "--plan-check-floor-ppm", "999999"}, &err) == 0, "floor 999999 is in range");
    check(take(scratch, {"--plan-check", expected_file, "--plan-check-floor-ppm", "0"}, &err) == 0, "floor 0 is in range");

    // the expected plan: unreadable is a file error (exit 3), malformed or not a v1 plan is invalid (exit 2)
    check(take(scratch, {"--plan-check", scratch + "/missing.json", "--plan-check-floor-ppm", ppm}, &err) == 3 &&
              err.find("BC_CFG_IO_FAILURE") != std::string::npos && err.find("bc-serve --plan-memory") != std::string::npos,
          "a missing expected plan is an io failure with the remedy: " + err);
    write(scratch + "/garbage.json", "{not json");
    check(take(scratch, {"--plan-check", scratch + "/garbage.json", "--plan-check-floor-ppm", ppm}, &err) == 2 &&
              err.find("BC_CFG_INVALID_FORMAT") != std::string::npos, "a malformed expected plan is refused: " + err);
    Json odd = prod->at("expected");
    odd["extra"] = 1;
    write(scratch + "/odd.json", odd.dump());
    check(take(scratch, {"--plan-check", scratch + "/odd.json", "--plan-check-floor-ppm", ppm}, &err) == 2 &&
              err.find("BC_CFG_UNKNOWN_FIELD") != std::string::npos && err.find("odd.json") != std::string::npos,
          "an expected plan with an unknown member is refused naming the file: " + err);

    // the check: admitted prints one line with its margins; a refusal is the shared error with its exit code
    check(take(scratch, {"--plan-check", expected_file, "--plan-check-floor-ppm", ppm}, &err) == 0 &&
              bc_plan::mode() == bc_plan::Mode::check, "a valid check is armed");
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

    std::fclose(stderr);
    if (failures) return 1;
    std::printf("ok\n");
    return 0;
}
