// Card-free producer/consumer test of the single Radiance option translation seam.
// The exact projection function called by rad0009 runs against all generated fields.
// A one-field table rename must change the emitted native argv.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "native_option_projection.h"
#include "serving_types.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cassert>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace v2 = bc::serving::v2;
using Json = nlohmann::json;

void check(bool condition, const std::string& what) {
    if (!condition)
        throw std::runtime_error(what);
}
} // namespace

int main() {
    try {
        size_t verified_options = 0;
        for (const auto& field : v2::kFields) {
            if (field.radiance_option.empty() || field.path.rfind("/spec/", 0) != 0)
                continue;
            Json doc = Json::object();
            const Json::json_pointer path{std::string(field.path)};
            if (field.radiance_switch)
                doc[path] = "on";
            else if (field.kind == "array")
                doc[path] = Json::array({"first", "second"});
            else if (field.kind == "integer" || field.kind == "ref")
                doc[path] = 47;
            else
                doc[path] = "value";

            const std::set<std::string> accepted{std::string(field.radiance_option)};
            const auto projection = v2::project_native_options(doc, v2::kFields, accepted);
            check(projection.error.empty(), std::string(field.path) + ": " + projection.error);
            check(projection.flags.size() == 1 && projection.flags.front().first == field.radiance_option,
                  std::string(field.path) + ": projected option not from table");
            const auto argv = projection.argv();
            check(argv.size() == (field.radiance_switch ? 1u : 2u) && argv.front() == field.radiance_option,
                  std::string(field.path) + ": emitted argv not from table");
            if (!field.radiance_switch)
                check(argv.back() == (field.kind == "array"                            ? "first,second"
                                      : field.kind == "integer" || field.kind == "ref" ? "47"
                                                                                       : "value"),
                      std::string(field.path) + ": wrong flag text");
            ++verified_options;
        }
        check(verified_options > 100, "registry option coverage unexpectedly dropped");

        Json explicit_zero = Json::object();
        explicit_zero["spec"]["processes"]["engine"]["tuning"]["prefill_interleave"] = 0;
        const auto zero = v2::project_native_options(explicit_zero, v2::kFields, {"--prefill-interleave"});
        check(zero.error.empty() && zero.argv() == std::vector<std::string>({"--prefill-interleave", "0"}),
              "explicit zero interleave was omitted from native argv");

        size_t env_count = 0;
        for (const auto& env : v2::kEnvBindings) {
            check(!env.path.empty() && !env.env.empty(), "unbound generated environment entry");
            for (const auto& field : v2::kFields)
                if (env.path == field.path)
                    check(field.radiance_option.empty(),
                          std::string(env.path) + ": option and environment both emitted");
            ++env_count;
        }
        check(env_count == v2::kEnvBindings.size() && env_count >= 12,
              "environment binding coverage unexpectedly dropped");

        auto renamed = v2::kFields;
        bool found = false;
        for (auto& field : renamed) {
            if (field.path != "/spec/processes/engine/server/port")
                continue;
            check(field.radiance_option == "--port", "unexpected port baseline");
            field.radiance_option = "--port-renamed-test";
            found = true;
        }
        check(found, "port table entry missing");
        Json doc = Json::object();
        doc["spec"]["processes"]["engine"]["server"]["port"] = 47001;
        const auto baseline = v2::project_native_options(doc, v2::kFields, {"--port"});
        const auto changed = v2::project_native_options(doc, renamed, {"--port-renamed-test"});
        check(baseline.error.empty() && changed.error.empty(), "port translation failed");
        check(baseline.argv() == std::vector<std::string>({"--port", "47001"}),
              "baseline argv no longer reflects registry");
        check(changed.argv() == std::vector<std::string>({"--port-renamed-test", "47001"}),
              "renaming a table entry did not rename the emitted argv");
        const auto wrong_parser = v2::project_native_options(doc, renamed, {"--port"});
        check(!wrong_parser.error.empty() && wrong_parser.flags.empty(),
              "stale native option silently accepted after rename");
        std::cout << "ok: " << verified_options << " option entries and " << env_count
                  << " env bindings, plus table rename to argv\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL native option projection: " << e.what() << '\n';
        return 1;
    }
}
