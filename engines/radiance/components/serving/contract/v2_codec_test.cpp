// Card-free v2 generated codec round-trip.
#include "../generated/v2/serving_codec.h"
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
        auto unknown = doc;
        unknown["spec"]["plan"]["power"]["unexpected"] = true;
        auto error = v2::decode(unknown, "", &decoded);
        if (!error || error->code != v2::ConfigErrorCode::unknown_field) {
            std::cerr << "v2 unknown property was not refused\n"; return 1;
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
