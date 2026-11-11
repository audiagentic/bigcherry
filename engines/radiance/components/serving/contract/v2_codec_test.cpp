// Card-free v2 generated codec round-trip; never linked by production before S13.
#include "../generated/v2/serving_codec.h"
#include "../generated/serving_codec.h"
#include "../v2/strict_json.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main() {
    namespace v2 = bc::serving::v2;

    for (const char* name : {"example.flashnext-4gpu.json", "example.lab-experiment.json"}) {
        std::ifstream in(std::string(BC_CONTRACT_V2_DOCS_DIR) + "/" + name);
        if (!in) { std::cerr << "missing fixture " << name << "\n"; return 1; }
        std::ostringstream ss;
        ss << in.rdbuf();
        const auto parsed = v2::parse_strict_json(ss.str(), name);
        if (!parsed) { std::cerr << "strict v2 JSON refused valid fixture\n"; return 1; }
        const auto doc = parsed.value();
        const auto duplicate = std::string("{\"apiVersion\":\"bigcherry/v2\",\"apiVersion\":\"bigcherry/v2\"}");
        const auto malformed = v2::parse_strict_json(duplicate, "duplicate-test");
        if (malformed || malformed.error().code != v2::ConfigErrorCode::duplicate_field) {
            std::cerr << "duplicate JSON property accepted\n"; return 1;
        }
        v2::Manifest decoded;
        if (auto e = v2::decode(doc, "", &decoded)) {
            std::cerr << "v2 fixture rejected " << name << ": " << e->path << " " << e->message << "\n";
            return 1;
        }
        if (v2::encode(decoded) != doc) { std::cerr << "roundtrip failed " << name << "\n"; return 1; }
        bc::serving::v1::Manifest old_version;
        const auto reverse = bc::serving::v1::decode(doc, "", &old_version);
        if (!reverse || reverse->code != bc::serving::v1::ConfigErrorCode::unsupported_version) {
            std::cerr << "v2 identity accepted by v1 decoder\n"; return 1;
        }
        auto unknown = doc;
        unknown["spec"]["plan"]["power"]["unexpected"] = true;
        auto error = v2::decode(unknown, "", &decoded);
        if (!error || error->code != v2::ConfigErrorCode::unknown_field) {
            std::cerr << "v2 unknown property was not refused\n"; return 1;
        }
        // A real committed v1 fixture must fail at the v2 identity boundary.
        std::ifstream old_in(std::string(BC_CONTRACT_V2_DOCS_DIR) +
                             "/../serving-manifest-v1/" + name);
        if (!old_in) { std::cerr << "missing v1 fixture " << name << "\\n"; return 1; }
        std::ostringstream old_ss;
        old_ss << old_in.rdbuf();
        const auto old_json = v2::parse_strict_json(old_ss.str(), name);
        if (!old_json) { std::cerr << "invalid v1 fixture JSON\\n"; return 1; }
        v2::Manifest old_as_v2;
        const auto version_error = v2::decode(old_json.value(), "", &old_as_v2);
        if (!version_error || version_error->code != v2::ConfigErrorCode::unsupported_version) {
            std::cerr << "real v1 fixture accepted by v2 decoder\\n"; return 1;
        }
        auto v1doc = doc;
        v1doc["apiVersion"] = "bigcherry/v1";
        error = v2::decode(v1doc, "", &decoded);
        if (!error || error->code != v2::ConfigErrorCode::unsupported_version) {
            std::cerr << "v1 identity accepted in v2 reader\n"; return 1;
        }
        auto bad = doc;
        bad["spec"]["plan"]["power"]["devices"][0]["settings"]["vddgfx_offset_mv"] = "wrong-type";
        if (!v2::decode(bad, "", &decoded)) {
            std::cerr << "non-integer power offset accepted\n"; return 1;
        }
    }
    std::cout << "v2 generated codec roundtrip/negative tests pass\n";
    return 0;
}
